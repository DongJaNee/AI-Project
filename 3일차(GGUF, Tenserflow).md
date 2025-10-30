<img width="360" height="257" alt="image" src="https://github.com/user-attachments/assets/ec336d7b-b331-42c2-b4b6-87c706dae73a" /><img width="308" height="167" alt="image" src="https://github.com/user-attachments/assets/86651317-4d78-4d24-a429-881a84d857be" />### GGUF
- 딥러닝 모델 파일 포맷으로 주로 LLM 추론 앱을 개발할 때 사용.
- 모델의 배포와 사용이 간편하며, On-Device AI와 같이 제한된 자원 환경에서도 좋은 추론 성능을 낼 수 있음.

**장점** 
- 빠른 응답 속도
- HW 호환성
- CPU/GPU 모두 지원 

### 1. 파인튜닝한 모델을 기동하기
- 허깅페이스 모델 repository에 저장하였던 모델을 다시 불러오기
```
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.55.4
!pip install --no-deps trl==0.22.2
```

```
from unsloth import FastLanguageModel
import torch

# 모델 이름 설정
model_name = "모델이름"  
max_seq_length=4096

# Unsloth를 사용하여 모델 로드
# 필요에 따라 max_seq_length, dtype, load_in_4bit 등의 인자를 조절하세요.
unsloth_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length, # 예시 값, 필요에 따라 조정
    dtype=None,          # 자동 감지 또는 필요에 따라 Float16, Bfloat16 설정
    load_in_4bit=True,   # 메모리 절약을 위해 4bit 양자화 로드 여부 설정
)

print(f"'{model_name}' 모델이 'unsloth_model' 변수로 로드되었습니다.")
```

### 2. HuggingFace 허브에 GGUF 모델 업로드
```
# Quantization 방식 설정
quantization_method = "q4_k_m"  # "f16" "q8_0" "q4_k_m" "q5_k_m"
```

```
# 업로드할 허깅페이스 repo 설정
huggingface_repo="seodongchan/abc"
```

```
# Hub 에 GGUF 업로드
unsloth_model.push_to_hub_gguf(
    huggingface_repo + "_GGUF",
    tokenizer,
    quantization_method=quantization_method,
    token= "hugging face write 토큰값" ,
)
```

```
from google.colab import drive
drive.mount('/content/drive')
```

#### 추가 gguf 파일 다운로드 
import gdown

```
# Google Drive 파일 ID 또는 전체 링크
google_drive_url = "https://drive.google.com/file/d/1qLVpQ93QgoP9cDo77eeg3PtRJh8VNbSr/view?usp=drive_link"

# 파일 다운로드
# gdown.download(google_drive_url, output="downloaded_file_name") # 파일 이름을 지정하려면 output 인자 사용
gdown.download(google_drive_url, quiet=False) # quiet=False 로 설정하면 다운로드 진행 상황을 볼 수 있습니다.

print("파일 다운로드가 완료되었습니다.")
```

## Ollama로 테스트하기 
### 1. Ollama Install

```
!curl -fsSL https://ollama.com/install.sh | sh
```

### 2. CMD 에서 Ollama Directory로 이동
#### Ollama List 
```
C:\Users\GKN>ollama list
```

<img width="711" height="96" alt="image" src="https://github.com/user-attachments/assets/45fd5b87-b5e9-41a6-ac0d-b8ad719d48ee" />

### 3. unsloth_model 생성
```
ollama create unsloth_model -f ./model/Modelfile
```

### 4. GGUF 파일 실행
```
ollama run unsloth.Q4_K_M.gguf
```

    




<img width="778" height="358" alt="image" src="https://github.com/user-attachments/assets/f76389cf-2fbb-4fb8-8463-5a669d65d2a5" />



<img width="708" height="305" alt="image" src="https://github.com/user-attachments/assets/8a014b3b-bb2a-4f88-a240-10a4f54ac827" />


---
## TensorFlowLite
- 개발자가 모바일, 내장형 기기, IoT 기기에서 모델을 실행할 수 있도록 지원하여 기기 내 AI모델을 사용할 수 있도록 하는 도구 모음
- 모바일 기기에 최적화
- Android 및 iOS기기, 내장형 Linux 및 마이크로 컨트롤러 등 지원

### 1. 안드로이드 스튜디오 설치
 Narwhal 4 download 

### 2. 가상 디바이스 구성 
1. More Actions→Virtual Device Manager를 클릭

 <img width="511" height="404" alt="image" src="https://github.com/user-attachments/assets/0a88b642-6b64-4e8b-9fdb-75a525bb6b39" />


2. Device Manager 창에 다음과 같이 Medium Phone API 36.1이 디폴트로 생성되어 있는 것을 확인

<img width="560" height="179" alt="image" src="https://github.com/user-attachments/assets/998bf5ea-54be-4f22-a583-956024946670" />


3. Virtual 오른쪽의 세모 버튼을 클릭하여 Medium Phone을 부팅시켜서 확인 

<img width="537" height="94" alt="image" src="https://github.com/user-attachments/assets/ffa5ab05-acd0-441d-b39d-85dbd49b3c6c" />


4. 오른쪽 끝에 세로로 나열된 아이콘 중 Device Manager 아이콘을클릭


<img width="290" height="288" alt="image" src="https://github.com/user-attachments/assets/ade00bad-0d7c-4598-ab27-6be7a1b3f85d" />


5. 디바이스 매니저 창에서 + 아이콘 Add a new Device를 클릭


<img width="316" height="135" alt="image" src="https://github.com/user-attachments/assets/35c28dc1-e669-4af2-8d7b-5fa6f691f846" />


6. Create Virtual Device를 클릭


<img width="308" height="167" alt="image" src="https://github.com/user-attachments/assets/23cb694c-7f54-4c9f-a538-1c0aad6b10e4" />


7.  Phone→ Pixel 7→Next를 선택


<img width="719" height="547" alt="image" src="https://github.com/user-attachments/assets/1ae85426-9c95-484f-9d06-043c45d9e720" />


### 5. 소스코드 다운로드

링크 : https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy9lYzIxOGNjODA5NjE4ZGZkL0ViM0pOY0NJTS1sTHZkd0x5SG1YenNZQnIzdkRJQzJUQjZCcjJUWlU4ZENNVVE%5FZT16U0VNY3M&cid=EC218CC809618DFD&id=EC218CC809618DFD%21sc035c9bd33884be9bddc0bc87997cec6&parId=EC218CC809618DFD%21s3e8f5abb8aa448fc81d137b7b06a5fa6&o=OneUp


### 4. 이미지 분류 앱 테스트 
1. 안드로이드 스튜디오를 기동

<img width="609" height="503" alt="image" src="https://github.com/user-attachments/assets/753dcb6e-1e94-4bbb-92a2-a43a8a788c61" />


2. 위의 화면에서 Open을 클릭하고 압축을 푼 최상위 디렉토리 tflite-imageclassifier-android를 선택하고 Select Folder 버튼을 클릭


<img width="360" height="257" alt="image" src="https://github.com/user-attachments/assets/a717d1e4-6c32-4244-8f80-f69f182f62f9" />

3. Device Manager edit

<img width="524" height="441" alt="image" src="https://github.com/user-attachments/assets/5ba5e198-99be-418b-9794-3916d375201c" />


4.  Camera에서 Rear를 Webcam0로 설정


<img width="877" height="666" alt="image" src="https://github.com/user-attachments/assets/6c3750bf-ffaf-4b82-a957-840d3960e7a1" />


5. 실행

<img width="509" height="67" alt="image" src="https://github.com/user-attachments/assets/ee52b63e-7baf-4796-ac80-12042ec690b9" />
