from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelConfig:
    def __init__(self):
        self.model_name = "facebook/opt-350m"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        try:
            # 입력 텍스트 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # 모델 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # 새로운 토큰 수 제한
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    no_repeat_ngram_size=3,  # 반복 방지
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 응답 디코딩
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 프롬프트 제거하고 응답만 반환
            response = response[len(prompt):].strip()
            
            # 응답이 비어있는 경우 처리
            if not response:
                return "No response generated."
                
            return response
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return f"Error generating response: {str(e)}"