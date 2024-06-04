import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


# load tokenizer
mistral_tokenizer = MistralTokenizer.from_file("/data/models/mistral_models/mistral_7b_instruct/tokenizer.model.v3")
# chat completion request
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
# encode message
tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
# load model
model = Transformer.from_folder("/data/models/mistral_models/mistral_7b_instruct/")
# generate results
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)
# decode generated tokens
result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
print(result)
