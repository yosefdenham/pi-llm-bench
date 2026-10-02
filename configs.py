from dataclasses import dataclass

@dataclass
class Config:
    model_family: str
    quant: str
    threads: int
    path: str
    params_b: float

def all_configs():
    models = [
        {"family": "tinyllama", "params_b": 1.1, "quant": "Q4_K_M",
         "path": "/home/ydenham/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
        {"family": "tinyllama", "params_b": 1.1, "quant": "Q5_K_M",
         "path": "/home/ydenham/models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"},
        {"family": "tinyllama", "params_b": 1.1, "quant": "Q6_K",
         "path": "/home/ydenham/models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf"},
        {"family": "tinyllama", "params_b": 1.1, "quant": "Q8_0",
         "path": "/home/ydenham/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"},
        {"family": "qwen2.5", "params_b": 1.5, "quant": "Q4_K_M",
         "path": "/home/ydenham/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
        {"family": "qwen2.5", "params_b": 1.5, "quant": "Q5_K_M",
         "path": "/home/ydenham/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"},
        {"family": "qwen2.5", "params_b": 1.5, "quant": "Q6_K",
         "path": "/home/ydenham/models/qwen2.5-1.5b-instruct-q6_k.gguf"},
        {"family": "qwen2.5", "params_b": 1.5, "quant": "Q8_0",
         "path": "/home/ydenham/models/qwen2.5-1.5b-instruct-q8_0.gguf"},
        {"family": "llama3.2", "params_b": 3.0, "quant": "Q4_K_M",
         "path": "/home/ydenham/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
        {"family": "llama3.2", "params_b": 3.0, "quant": "Q5_K_M",
         "path": "/home/ydenham/models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"},
        {"family": "llama3.2", "params_b": 3.0, "quant": "Q6_K",
         "path": "/home/ydenham/models/Llama-3.2-3B-Instruct-Q6_K.gguf"},
        {"family": "llama3.2", "params_b": 3.0, "quant": "Q8_0",
         "path": "/home/ydenham/models/Llama-3.2-3B-Instruct-Q8_0.gguf"},
    ]
    configs = []
    for m in models:
        for t in [1, 2, 3, 4]:
            configs.append(Config(
    	        model_family=m["family"],
                quant=m["quant"],
                threads=t,
                path=m["path"],
                params_b=m["params_b"],
            ))
    return configs
