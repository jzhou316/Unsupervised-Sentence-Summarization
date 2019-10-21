"""
Sequentially embed tokens into ELMo vectors, using only forward computation, with externally updated hidden states.

Based on allennlp.commands.elmo.ElmoEmbedder and allennlp.modules.elmo._ElmoBiLm.
"""

import json
import logging
from typing import List, Iterable, Tuple, Any, Optional, Dict
import warnings

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     import h5py
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

import numpy
import torch
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.common.checks import ConfigurationError
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from allennlp.modules.elmo import batch_to_ids, _ElmoCharacterEncoder

from elmo_lstm_forward import ElmoLstmForward

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


class ElmoEmbedderForward(torch.nn.Module):
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None,
                 cuda_device: int = -1) -> None:
        super(ElmoEmbedderForward, self).__init__()
        
        self._token_embedder = _ElmoCharacterEncoder2(options_file, weight_file, requires_grad=requires_grad)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)
            self.vocab = vocab_to_cache                 # the first token should be the padding token, with id = 0

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        
        logger.info("Initializing ELMo Forward.")
        self._elmo_lstm_forward = ElmoLstmForward(input_size=options['lstm']['projection_dim'],
                                                  hidden_size=options['lstm']['projection_dim'],
                                                  cell_size=options['lstm']['dim'],
                                                  num_layers=options['lstm']['n_layers'],
                                                  memory_cell_clip_value=options['lstm']['cell_clip'],
                                                  state_projection_clip_value=options['lstm']['proj_clip'],
                                                  requires_grad=requires_grad)
        self._elmo_lstm_forward.load_weights(weight_file)
        if cuda_device >= 0:
            self._elmo_lstm_forward = self._elmo_lstm_forward.cuda(device=cuda_device)
            self._token_embedder = self._token_embedder.cuda(device=cuda_device)
#            self.cuda(device=cuda_device)                   # this happens in-place
        self.cuda_device = cuda_device if cuda_device >= 0 else 'cpu'
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1
        
    def batch_to_embeddings(self,
                            batch: List[List[str]],
                            add_bos: bool = False,
                            add_eos: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sentence insensitive token representations for a batch of tokenized sentences,
        using pretrained character level CNN. This is the first layer of ELMo representation.
        
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        add_bos: ``bool``
            Whether to add begin of sentence token.
        add_eos: ``bool``
            Whether to add end of sentence token.
        Returns
        -------
        type_representation: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 0/1/2, embedding_dim)`` tensor with context
            insensitive token representations.
        mask: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 0/1/2)`` long tensor with sequence mask.
        """
        
        if self._word_embedding is not None:          # vocab_to_cache was passed in the constructor of this class
            try:
                word_inputs = [[self.vocab.index(w) for w in b] for b in batch]
                max_timesteps = max([len(b) for b in word_inputs])
                word_inputs = [b + [0] * (max_timesteps - len(b)) if len(b) < max_timesteps else b
                               for b in word_inputs]      # 0 is the padding id
                word_inputs = torch.tensor(word_inputs, dtype=torch.long, device=self.cuda_device)
                                                          # word ids in the cached vocabulary
                                                          # LongTensor of shape (batch_size, max_timesteps)
                
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 0/1/2, embedding_dim)
                type_representation, mask = add_sentence_boundaries(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding,
                        add_bos,
                        add_eos
                )
            except RuntimeError:
                character_ids = batch_to_ids(batch)           # size (batch_size, max_timesteps, 50)
                if self.cuda_device >= 0:
                    character_ids = character_ids.cuda(device=self.cuda_device)
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(character_ids, add_bos, add_eos)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            character_ids = batch_to_ids(batch)           # size (batch_size, max_timesteps, 50)
            if self.cuda_device >= 0:
                character_ids = character_ids.cuda(device=self.cuda_device)
            token_embedding = self._token_embedder(character_ids, add_bos, add_eos)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
            
        return type_representation, mask
        
    def forward(self,
                batch: List[List[str]],
                add_bos: bool = False,
                add_eos: bool = False,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[List[numpy.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        add_bos: ``bool``
            Whether to add begin of sentence token.
        add_eos: ``bool``
            Whether to add end of sentence token.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 1 * hidden_size) and
            (num_layers, batch_size, 1 * cell_size) respectively.
            
            Or, with shape (num_layers, 1 * hidden_size) and
            (num_layers, 1 * cell_size) respectively, if all the batch share the same initial_state.
            
        Returns
        -------
        lstm_outputs : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states : ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 1 * hidden_size) and  (num_layers, batch_size, 1 * cell_size)
            respectively. The last dimension is NOT duplicated because it contains the state/memory
            for ONLY the forward layers.
            
        elmo_embeddings: ``list[numpy.ndarray]``
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        batch_size = len(batch)
        if initial_state is not None:                        # TO DO: need to deal with changing batch size
            initial_state_shape = list(initial_state[0].size())
            if len(initial_state_shape) == 2:
                initial_state = (initial_state[0].expand(batch_size, -1, -1).transpose(0, 1),
                                 initial_state[1].expand(batch_size, -1, -1).transpose(0, 1))
            elif len(initial_state_shape) == 3:
                pass
            else:
                raise ValueError("initial_state only accepts tuple of 2D or 3D input")
        
        token_embedding, mask = self.batch_to_embeddings(batch, add_bos, add_eos)
        lstm_outputs, final_states = self._elmo_lstm_forward(token_embedding, mask, initial_state)
        
        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        
        output_tensors = [token_embedding * mask.float().unsqueeze(-1)]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))
        
        # without_bos_eos is a 3 element list of tuples of (batch_size, num_timesteps, dim) and 
        # (batch_size, num_timesteps) tensors, each element representing a layer.
        without_bos_eos = [remove_sentence_boundaries(layer, mask, add_bos, add_eos)
                           for layer in output_tensors]
        # Split the list of tuples into two tuples, each of length 3
        activations_without_bos_eos, mask_without_bos_eos = zip(*without_bos_eos)
        
        # Convert the activations_without_bos_eos into a single batch first tensor,
        # of size (batch_size, num_layers, num_timesteps, dim)
        activations = torch.cat([ele.unsqueeze(1) for ele in activations_without_bos_eos], dim=1)
        # The mask is the same for each ELMo layer, so just take the first.
        mask_without_bos_eos = mask_without_bos_eos[0]
        
        # organize the Elmo embeddings into a list corresponding to the batch of sentences
        elmo_embeddings = []
        for i in range(batch_size):
            length = int(mask_without_bos_eos[i, :].sum())
            if length == 0:
                raise ConfigurationError('There exists totally masked out sequence in the batch.')
            else:
#                 elmo_embeddings.append(activations[i, :, :length, :].detach().cpu().numpy())
                elmo_embeddings.append(activations[i, :, :length, :].detach())
        
        return elmo_embeddings, final_states
    
    def embed_sentence(self,
                       sentence: List[str],
                       add_bos: bool = False,
                       add_eos: bool = False,
                       initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
                                    Tuple[numpy.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the forward only ELMo embeddings for a single tokenized sentence.
        See the comment under the class definition.
        Parameters
        ----------
        sentence : ``List[str]``, required
            A tokenized sentence.
        add_bos: ``bool``
            Whether to add begin of sentence token.
        add_eos: ``bool``
            Whether to add end of sentence token.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, 1, 1 * hidden_size) and
            (num_layers, 1, 1 * cell_size) respectively.
            
            Or, with shape (num_layers, 1 * hidden_size) and
            (num_layers, 1 * cell_size) respectively.
        Returns
        -------
        A tensor containing the ELMo vectors, and 
        final states, tuple of size (num_layers, hidden_size) and (num_layers, memory_size).
        """
        elmo_embeddings, final_states = self.forward([sentence], add_bos, add_eos, initial_state)

        return elmo_embeddings[0], tuple([ele.squeeze(1) for ele in final_states])
    
    def embed_sentences(self,
                        sentences: Iterable[List[str]],
                        add_bos: bool = False,
                        add_eos: bool = False,
                        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        batch_size: int = DEFAULT_BATCH_SIZE) -> \
                    List[Tuple[numpy.ndarray, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Computes the forward only ELMo embeddings for a iterable of sentences.
        See the comment under the class definition.
        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        add_bos: ``bool``
            Whether to add begin of sentence token.
        add_eos: ``bool``
            Whether to add end of sentence token.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 1 * hidden_size) and
            (num_layers, batch_size, 1 * cell_size) respectively.
            
            Or, with shape (num_layers, 1 * hidden_size) and
            (num_layers, 1 * cell_size) respectively, if all the batch share the same initial_state.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.
        Returns
        -------
            A list of tuple of (numpy.ndarray/torch.Tensor, (torch.Tensor, torch.Tensor)), 
            each representing the ELMo vectors for the input sentence 
            at the same index, and the final states after running that sentence, with shape (num_layers, hidden_size) and
            (num_layers, cell_size) respectively.
            (The return type could also be a generator. Can convert to a list using list().)
        """
        embeddings_and_states = []
        print('Embedding sentences into forward ELMo vectors ---')
#         for batch in Tqdm.tqdm(lazy_groups_of(iter(sentences), batch_size)):
        for batch in lazy_groups_of(iter(sentences), batch_size):
            elmo_embeddings, final_states = self.forward(batch, add_bos, add_eos, initial_state)
            # Remember: final_states is a tuple of tensors
            final_states_chunked = []
            for i in range(2):
                final_states_chunked.append(list(map(lambda x: torch.squeeze(x, dim=1),
                                                     final_states[i].chunk(final_states[i].size(1), dim=1))))
            final_states_chunked = list(zip(*final_states_chunked))
            assert len(elmo_embeddings) == len(final_states_chunked), 'length of embeddings and final states mismatch'
#            yield from zip(elmo_embeddings, final_states_chunked)
            embeddings_and_states += list(zip(elmo_embeddings, final_states_chunked))
        return embeddings_and_states
    
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        """
        Given a list of tokens, this method precomputes word representations
        by running just the character convolutions and highway layers of elmo,
        essentially creating uncontextual word vectors. On subsequent forward passes,
        the word ids are looked up from an embedding, rather than being computed on
        the fly via the CNN encoder.
        This function sets 3 attributes:
        _word_embedding : ``torch.Tensor``
            The word embedding for each word in the tokens passed to this method.
        _bos_embedding : ``torch.Tensor``
            The embedding for the BOS token.
        _eos_embedding : ``torch.Tensor``
            The embedding for the EOS token.
        Parameters
        ----------
        tokens : ``List[str]``, required.
            A list of tokens to precompute character convolutions for.
        """
        tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
        timesteps = 32
        batch_size = 32
        chunked_tokens = lazy_groups_of(iter(tokens), timesteps)

        all_embeddings = []
        device = get_device_of(next(self.parameters()))
        for batch in lazy_groups_of(chunked_tokens, batch_size):
            # Shape (batch_size, timesteps, 50)
            batched_tensor = batch_to_ids(batch)
            # NOTE: This device check is for when a user calls this method having
            # already placed the model on a device. If this is called in the
            # constructor, it will probably happen on the CPU. This isn't too bad,
            # because it's only a few convolutions and will likely be very fast.
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._token_embedder(batched_tensor, add_bos=False, add_eos=False)
            token_embedding = output["token_embedding"]
            mask = output["mask"]
            token_embedding, _ = remove_sentence_boundaries(token_embedding, mask, rmv_bos=False, rmv_eos=False)
            all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
        full_embedding = torch.cat(all_embeddings, 0)

        # We might have some trailing embeddings from padding in the batch, so
        # we clip the embedding and lookup to the right size.
        full_embedding = full_embedding[:len(tokens), :]
        embedding = full_embedding[2:len(tokens), :]
        vocab_size, embedding_dim = list(embedding.size())

        from allennlp.modules.token_embedders import Embedding # type: ignore
        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(vocab_size, # type: ignore
                                         embedding_dim,
                                         weight=embedding.data,
                                         trainable=self._requires_grad,
                                         padding_index=0)

        
class _ElmoCharacterEncoder2(_ElmoCharacterEncoder):
    @overrides
    def forward(self,
                inputs: torch.Tensor,
                add_bos: bool = False,
                add_eos: bool = False) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.
        add_bos: ``bool``
            Whether to add begin of sentence symbol
        add_eos: ``bool``
            Whether to add end of sentence symbol
        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 0/1/2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 0/1/2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS (this is the only difference from the original _ElmoCharacterEncoder class)
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundaries(
                inputs,
                mask,
                self._beginning_of_sentence_characters,
                self._end_of_sentence_characters,
                add_bos,
                add_eos
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
                character_ids_with_bos_eos.view(-1, max_chars_per_token),
                self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
                'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }
        

def add_sentence_boundaries(tensor: torch.Tensor,
                            mask: torch.Tensor,
                            sentence_begin_token: Any,
                            sentence_end_token: Any,
                            add_bos: bool = False,
                            add_eos: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 0/1/2)`` or ``(batch_size, timesteps + 0/1/2, dim)`` respectively.
    Returns both the new tensor and updated mask.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)`` (assuming padding id is always 0)
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.
    add_bos: bool
        Whether to add begin of sentence token.
    add_eos: bool
        Whether to add end of sentence token.
    Returns
    -------
    tensor_with_boundary_tokens : ``torch.Tensor``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 0/1/2) and if the input was 3D, it has shape
        (batch_size, timesteps + 0/1/2, dim).
    new_mask : ``torch.Tensor``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    if add_bos:
        new_shape[1] = new_shape[1] + 1
    if add_eos:
        new_shape[1] = new_shape[1] + 1
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
    if len(tensor_shape) == 2:
        if add_bos:
            tensor_with_boundary_tokens[:, 1:(1 + tensor_shape[1])] = tensor
            tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        else:
            tensor_with_boundary_tokens[:, 0:tensor_shape[1]] = tensor
        if add_eos:
            for i, j in enumerate(sequence_lengths):
                tensor_with_boundary_tokens[i, j + 1 if add_bos else j] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens != 0).long()
    elif len(tensor_shape) == 3:
        if add_bos:
            tensor_with_boundary_tokens[:, 1:(1 + tensor_shape[1]), :] = tensor
        else:
            tensor_with_boundary_tokens[:, 0:tensor_shape[1], :] = tensor
        for i, j in enumerate(sequence_lengths):
            if add_bos:
                tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            if add_eos:
                tensor_with_boundary_tokens[i, j + 1 if add_bos else j, :] = sentence_end_token
        new_mask = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask

def remove_sentence_boundaries(tensor: torch.Tensor,
                               mask: torch.Tensor,
                               rmv_bos: bool = False,
                               rmv_eos: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape ``(batch_size, timesteps - 0/1/2)`` or
    ``(batch_size, timesteps - 0/1/2, dim)`` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
    to be 1).
    Returns both the new tensor and updated mask.
    This function is the inverse of ``add_sentence_boundaries``.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    rmv_bos: bool
        Whether to remove begin of sentence token
    rmv_eos: bool
        Whether to remove end of sentence token
    Returns
    -------
    tensor_without_boundary_tokens : ``torch.Tensor``
        The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 0/1/2)``
        or ``(batch_size, timesteps - 0/1/2, dim)``
    new_mask : ``torch.Tensor``
        The new mask for the tensor of shape ``(batch_size, timesteps - 0/1/2)``.
    """
    # TODO: matthewp, profile this transfer
    if not rmv_bos and not rmv_eos:
        return tensor, mask
    
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    if rmv_bos:
        new_shape[1] = new_shape[1] - 1
    if rmv_eos:
        new_shape[1] = new_shape[1] - 1
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if rmv_bos and rmv_eos and j > 2:
            if len(tensor_shape) == 3:
                tensor_without_boundary_tokens[i, :(j - 2), :] = tensor[i, 1:(j - 1), :]
            elif len(tensor_shape) == 2:
                tensor_without_boundary_tokens[i, :(j - 2)] = tensor[i, 1:(j - 1)]
            else:
                raise ValueError("remove_sentence_boundaries only accepts 2D and 3D input")
            new_mask[i, :(j - 2)] = 1
        if rmv_bos and not rmv_eos and j > 1:
            if len(tensor_shape) == 3:
                tensor_without_boundary_tokens[i, :(j - 1), :] = tensor[i, 1:j, :]
            elif len(tensor_shape) == 2:
                tensor_without_boundary_tokens[i, :(j - 1)] = tensor[i, 1:j]
            else:
                raise ValueError("remove_sentence_boundaries only accepts 2D and 3D input")
            new_mask[i, :(j - 1)] = 1
        if not rmv_bos and rmv_eos and j > 1:
            if len(tensor_shape) == 3:
                tensor_without_boundary_tokens[i, :(j - 1), :] = tensor[i, :(j - 1), :]
            elif len(tensor_shape) == 2:
                tensor_without_boundary_tokens[i, :(j - 1)] = tensor[i, :(j - 1)]
            else:
                raise ValueError("remove_sentence_boundaries only accepts 2D and 3D input")
            new_mask[i, :(j - 1)] = 1

    return tensor_without_boundary_tokens, new_mask        
        
