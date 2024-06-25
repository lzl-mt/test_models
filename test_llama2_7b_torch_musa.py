import torch
import torch_musa
import time
from numpy import percentile
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_dir = '/data/models/llama-2-7b-chat-hf-fp16/'
batch_size = 48
num_tokens = 256

tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="musa",
                                          trust_remote_code=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="musa",
                                             trust_remote_code=True, torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained(model_dir)

device = 'musa'
print('#----------------------------warmup start---------------------------')
with torch.no_grad():
    # prompts = "春眠不觉晓，处处闻啼鸟。"
    prompts = "[Round 1]\n\n问：如何获得同事的认可\n\n答：\n"
    input_ids = tokenizer(prompts).input_ids
    # print(f"input: {tokenizer.decode(input_ids)}")
    input_ids = torch.LongTensor([input_ids]).to(device)
    # print(input_ids.shape)

    output_ids = list()
    past_key_values = None

    output = model(input_ids, use_cache=True, past_key_values=past_key_values)
    logits = output.logits
    past_key_values = output.past_key_values
    res = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    output_ids.append(int(res.cpu().numpy()[0][0]))
    input_ids = res

    for i in range(256):
        output = model(input_ids, use_cache=True,
                       past_key_values=past_key_values)
        logits = output.logits
        past_key_values = output.past_key_values
        res = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_ids.append(int(res.cpu().numpy()[0][0]))
        input_ids = res
    # print(f"output_ids: {output_ids}")
    print("output:", tokenizer.decode(output_ids).strip())
print('#----------------------------warmup end -----------------------------')

# ----------------------------profile start---------------------------
# for i in [128, 256, 512, 1024, 2048, 3072]:
st = time.time()
prefill_time = 0
decode_time = 0
with torch.no_grad():
    # generate random input_ids.
    input_ids = torch.randint(
        10240, (batch_size, num_tokens)).to(device) + 10240

    # generate by multiple prompts.
    # prompts = ["[Round 1]\n\n问：如何获得同事的认可\n\n答：\n" for i in range(batch_size)]
    # input_ids = tokenizer(prompts).input_ids
    # input_ids = torch.LongTensor(input_ids).to(device)

    st = time.time()
    output_ids = torch.LongTensor()
    # output_ids = list()
    past_key_values = None

    # start prefill
    prefill_start = time.time()
    output = model(input_ids, use_cache=True, past_key_values=past_key_values)
    logits = output.logits
    past_key_values = output.past_key_values
    res = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    output_ids = torch.cat((output_ids, res.cpu()), dim=1)
    input_ids = res
    prefill_end = time.time()
    prefill_time = prefill_end - prefill_start

    # start generate tokens
    for i in range(num_tokens):
        output = model(input_ids, use_cache=True,
                       past_key_values=past_key_values)
        logits = output.logits
        past_key_values = output.past_key_values
        res = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_ids = torch.cat((output_ids, res.cpu()), dim=1)
        input_ids = res

    decode_time = time.time() - prefill_end
    # print the resultes
    # for output_id in output_ids:
    #     print(output_id)
    #     print(f"output: {tokenizer.decode(output_id.tolist())}")

consume_time = time.time() - st

num_tokens = batch_size * num_tokens
fps = num_tokens / decode_time
print(f'generate token fps: {fps:.3f} tokens/s')
print(
    f'prefill latency :{prefill_time:.3f} s, single one batch prefill latency:{prefill_time/batch_size:.3f} s')
print(
    f'decode latency :{decode_time:.3f} s, single one batch decode latency:{decode_time/batch_size:.3f} s')
print(f'end-to-end time:{consume_time:.3f} s,  num_tokens:{num_tokens}')