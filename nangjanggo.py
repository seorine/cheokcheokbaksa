import json
import os
from datetime import datetime, timedelta

# scikit-learn, joblib 같은 외부 라이브러리 필요
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------
# 파일/폴더 경로 설정
# -----------------------------
DATA_DIR = "data"  # 모든 데이터(JSON, 모델)를 저장할 폴더
INGR_FILE = os.path.join(DATA_DIR, "ingredients.json")  # 식재료 목록
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")  # 레시피 피드백(좋아요/싫어요)
MODEL_FILE = os.path.join(DATA_DIR, "recipe_model.pkl")  # 학습된 AI 모델 파일

# -----------------------------------
# 유틸 함수: 폴더/파일 관리
# -----------------------------------
def ensure_data_dir():
    """
    data 폴더랑 기본 JSON 파일들을 준비해 두는 함수.
    - 프로그램을 처음 돌릴 때 자동으로 폴더/파일 생성
    - 이미 있으면 건드리지 않음
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 냉장고 식재료 목록 파일이 없으면 빈 리스트로 생성
    if not os.path.exists(INGR_FILE):
        with open(INGR_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # 레시피 피드백 파일이 없으면 빈 리스트로 생성
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def load_json(path):
    """
    JSON 파일을 파이썬 객체(list/dict)로 읽어오는 공통 함수.
    파일이 없으면 그냥 빈 리스트 반환.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    """
    파이썬 객체를 JSON 파일로 저장하는 공통 함수.
    indent=2 로 저장해서 사람이 봐도 읽기 편함.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -----------------------------------
# 냉장고 매니저: CRUD + 유통기한/보관 팁
# -----------------------------------
class FridgeManager:
    """
    냉장고 안의 '식재료 상태'를 책임지는 클래스.
    - JSON 파일에서 읽어 오고
    - 메모리에서 수정하고
    - 다시 JSON 파일에 저장하는 역할
    """

    def __init__(self):
        # 데이터 폴더/파일이 준비되어 있는지 확인
        ensure_data_dir()
        # 기존에 저장된 식재료 목록을 전부 메모리에 로딩
        self.ingredients = load_json(INGR_FILE)

    def save(self):
        """
        현재 메모리에 있는 self.ingredients 전체를 JSON 파일로 저장.
        프로그램 껐다 켜도 이 파일 덕분에 데이터가 유지됨.
        """
        save_json(INGR_FILE, self.ingredients)

    def add_ingredient(self, name, quantity, category, location, expiry_str):
        """
        새로운 식재료 1개를 추가하는 함수.
        - name: 이름 (예: '계란')
        - quantity: 수량 (정수)
        - category: 채소/고기/유제품/기타
        - location: 냉장/냉동/실온
        - expiry_str: 'YYYY-MM-DD' 형식의 유통기한 문자열
        """
        item = {
            "name": name,
            "quantity": quantity,
            "category": category,
            "location": location,
            "expiry": expiry_str
        }
        self.ingredients.append(item)
        self.save()  # 추가 후 바로 파일에 저장 (데이터 유실 방지)

        print(f"[+] 추가됨: {item}")

    def list_ingredients(self):
        """
        현재 냉장고에 들어 있는 모든 식재료를 번호와 함께 출력.
        발표 때 데모하기 좋은 함수.
        """
        if not self.ingredients:
            print("냉장고가 비어 있습니다.")
            return

        for i, item in enumerate(self.ingredients, start=1):
            print(
                f"{i}. {item['name']} | 수량: {item['quantity']} | "
                f"분류: {item['category']} | 보관: {item['location']} | "
                f"유통기한: {item['expiry']}"
            )

    def get_expiring(self, days=3):
        """
        앞으로 days일 안에 유통기한이 끝나는 식재료 목록만 골라서 반환.
        - 내부 로직만 하고, 출력은 다른 함수가 담당.
        """
        today = datetime.today().date()
        limit = today + timedelta(days=days)
        result = []

        for item in self.ingredients:
            try:
                exp = datetime.strptime(item["expiry"], "%Y-%m-%d").date()
            except ValueError:
                # 날짜 형식이 잘못되어 있으면 그냥 스킵
                continue

            if today <= exp <= limit:
                result.append(item)

        return result

    def print_expiring_alert(self, days=3):
        """
        get_expiring() 결과를 예쁘게 출력하는 함수.
        → 발표 때 '유통기한 임박 알림' 기능 데모용.
        """
        expiring = self.get_expiring(days)
        if not expiring:
            print(f"앞으로 {days}일 안에 유통기한이 끝나는 식재료가 없습니다.")
            return

        print(f"[유통기한 임박 식재료 ({days}일 이내)]")
        for item in expiring:
            print(f"- {item['name']} ({item['expiry']})")

    # [NEW]
    def get_expired(self):
        """
        이미 유통기한이 지난 식재료 목록만 반환.
        (오늘 날짜 기준 exp < today)
        """
        today = datetime.today().date()
        result = []

        for item in self.ingredients:
            try:
                exp = datetime.strptime(item["expiry"], "%Y-%m-%d").date()
            except ValueError:
                # 날짜 형식이 잘못되어 있으면 그냥 스킵
                continue

            if exp < today:
                result.append((item, (today - exp).days))  # (식재료, 며칠 지남)

        return result

    # [NEW]
    def print_expired_alert(self):
        """
        이미 유통기한이 지난 식재료들을 출력해 주는 함수.
        """
        expired = self.get_expired()
        if not expired:
            print("유통기한이 지난 식재료가 없습니다.")
            return

        print("[유통기한 지난 식재료]")
        for item, days_over in expired:
            print(f"- {item['name']} ({item['expiry']}) | {days_over}일 지남")

    def get_storage_tip(self, item):
        """
        아주 단순한 규칙 기반 보관 팁.
        나중에 이 부분을 'AI로 대체할 수 있다'는 포인트를 발표 때 강조하기 좋음.
        """
        cat = item["category"]
        if cat == "채소":
            return "채소는 밀폐용기에 보관하고, 냉장 보관하세요."
        elif cat == "고기":
            return "고기는 0~4도에서 냉장, 장기 보관은 냉동하세요."
        elif cat == "유제품":
            return "유제품은 온도 변화가 적은 안쪽 선반에 두는 것이 좋습니다."
        else:
            return "각 제품의 포장지에 있는 보관 방법을 참고하세요."

    # [NEW]
    def remove_ingredient(self, index_1based):
        """
        번호(1부터 시작)를 받아서 해당 식재료를 삭제하는 함수.
        """
        idx = index_1based - 1
        if 0 <= idx < len(self.ingredients):
            removed = self.ingredients.pop(idx)
            self.save()
            print(f"[-] 삭제됨: {removed['name']} (유통기한: {removed['expiry']})")
        else:
            print("올바르지 않은 번호입니다.")

# -----------------------------------
# AI 레시피 추천 모델
# - 피드백 기반 간단한 선호도 학습
# -----------------------------------
class RecipeAI:
    """
    사용자가 남긴 레시피 피드백(좋아요/싫어요)을 이용해서
    '이 재료 조합을 사용한 레시피를 좋아할 확률'을 추정하는 AI 담당 클래스.

    여기서 핵심 포인트:
    - 텍스트(재료 이름들)를 벡터로 변환 (CountVectorizer)
    - 선형 모델(SGDClassifier)에 학습
    - 학습 결과는 recipe_model.pkl로 저장 → 다음 실행에서도 재사용 가능
    """

    def __init__(self):
        ensure_data_dir()
        # 과거에 쌓인 피드백(좋아요/싫어요) 로딩
        self.feedback = load_json(FEEDBACK_FILE)
        self.model = None  # 아직 학습 안 된 상태

    def add_feedback(self, ingredients_text, liked: bool):
        """
        레시피 피드백 1개를 추가.
        - ingredients_text: "닭고기 양파 마늘" 식으로 재료를 공백으로 나열
        - liked: 사용자가 이 레시피를 좋아했는지 (True/False)
        """
        entry = {
            "ingredients": ingredients_text,
            "liked": 1 if liked else 0  # 분류 모델을 위해 1/0 으로 저장
        }
        self.feedback.append(entry)
        # 피드백도 JSON 파일에 즉시 저장 → 데이터 누적
        save_json(FEEDBACK_FILE, self.feedback)

        print(f"[피드백 저장] {entry}")

    def train_model(self):
        """
        지금까지 쌓인 피드백을 바탕으로 AI 모델을 학습.
        - 피드백이 3개 미만이면 학습하지 않고 경고만 출력.
        """
        if len(self.feedback) < 3:
            print("학습 데이터가 너무 적습니다. (3개 이상 필요)")
            return

        # 텍스트(재료 리스트)와 라벨(좋아요/싫어요)을 분리
        texts = [fb["ingredients"] for fb in self.feedback]
        labels = [fb["liked"] for fb in self.feedback]

        # 파이프라인:
        # 1) CountVectorizer: 텍스트 → Bag-of-Words 벡터
        # 2) SGDClassifier: 확률 출력 가능한 로지스틱 회귀 형태 (log_loss)
        pipeline = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", SGDClassifier(loss="log_loss"))
        ])

        pipeline.fit(texts, labels)
        self.model = pipeline

        # 학습된 모델을 파일로 저장 → 다음 실행에서도 다시 학습할 필요 X
        joblib.dump(self.model, MODEL_FILE)
        print("[모델 학습 완료] 저장:", MODEL_FILE)

    def load_model(self):
        """
        저장된 모델 파일(recipe_model.pkl)이 있으면 로딩.
        없으면 model은 None 상태로 둔다.
        """
        if os.path.exists(MODEL_FILE):
            self.model = joblib.load(MODEL_FILE)
        else:
            self.model = None

    def score_recipe(self, ingredients_text):
        """
        주어진 재료 조합에 대해
        '사용자가 좋아할 확률(0~1 사이)'을 추정해서 반환.
        """
        if self.model is None:
            # 아직 메모리에 모델이 없으면, 먼저 파일에서 로딩 시도
            self.load_model()

        if self.model is None:
            # 파일도 없으면, 아직 학습된 모델이 없는 상태 → 중간값 0.5 반환
            return 0.5

        prob = self.model.predict_proba([ingredients_text])[0][1]
        return float(prob)

    def suggest_recipes(self, fridge_items):
        """
        냉장고 안 재료들을 기반으로 '가상의 레시피 후보' 세 개를 만들고
        각 후보에 대해 선호도 점수를 매긴 뒤, 높은 순으로 정렬해서 반환.

        실제 구현에서는 여기서 진짜 레시피 데이터베이스를 붙일 수 있음.
        지금은 'AI 구조를 보여주는 데모용'으로 간단하게만 처리.
        """
        # 냉장고에 있는 재료 이름들을 하나의 문자열로 합치기
        base_names = [item["name"] for item in fridge_items]
        base_text = " ".join(base_names)

        # 세 가지 가상의 레시피 후보 정의
        candidates = [
            {"name": "볶음 요리", "need": base_text},
            {"name": "국/찌개 요리", "need": base_text + " 물"},
            {"name": "샐러드", "need": base_text + " 채소 올리브유"}
        ]

        result = []
        for c in candidates:
            score = self.score_recipe(c["need"])
            result.append({**c, "score": score})

        # 점수가 높은 순으로 정렬해서 '추천 순위'처럼 보여주기
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

# -----------------------------------
# CLI 메뉴(텍스트 기반 인터페이스)
# -----------------------------------
def main_cli():
    """
    실제로 사용자가 터미널/콘솔에서 만지는 부분.
    - 메뉴 숫자를 입력해서 기능을 선택
    - 내부적으로는 FridgeManager + RecipeAI 를 호출
    """
    fridge = FridgeManager()
    ai = RecipeAI()

    while True:
        print("\n==== AI 냉장고 매니저 ====")
        print("1. 식재료 추가")
        print("2. 식재료 목록 보기")
        print("3. 유통기한 임박 알림 보기")
        print("4. 보관 방법 추천 보기")
        print("5. 레시피 추천 (AI)")
        print("6. 레시피 피드백 저장 (좋아요/싫어요)")
        print("7. AI 모델 학습하기")
        print("8. 유통기한 지난 식재료 보기")   # [NEW]
        print("9. 식재료 삭제")               # [NEW]
        print("0. 종료")
        cmd = input("메뉴 선택: ").strip()

        if cmd == "1":
            # 새 식재료를 입력 받아서 추가
            name = input("이름: ")
            quantity = int(input("수량: "))
            category = input("분류(채소/고기/유제품/기타): ")
            location = input("보관(냉장/냉동/실온): ")
            expiry = input("유통기한 (예: 2025-11-21): ")
            fridge.add_ingredient(name, quantity, category, location, expiry)

        elif cmd == "2":
            # 현재 냉장고 전체 목록 출력
            fridge.list_ingredients()

        elif cmd == "3":
            # 특정 기간 안에 유통기한이 임박한 재료만 보여주기
            days_str = input("며칠 이내 식재료를 볼까요? (기본 3): ").strip()
            days = int(days_str) if days_str else 3
            fridge.print_expiring_alert(days)

        elif cmd == "4":
            # 특정 재료 하나를 골라서 보관 팁 보여주기
            fridge.list_ingredients()
            if not fridge.ingredients:
                continue
            idx = int(input("보관 방법을 알고 싶은 번호: "))
            if 1 <= idx <= len(fridge.ingredients):
                item = fridge.ingredients[idx - 1]
                tip = fridge.get_storage_tip(item)
                print(f"[{item['name']}] 보관 팁: {tip}")
            else:
                print("번호가 올바르지 않습니다.")

        elif cmd == "5":
            # AI 레시피 추천 호출
            if not fridge.ingredients:
                print("냉장고가 비어 있어서 추천할 수 없습니다.")
                continue

            recipes = ai.suggest_recipes(fridge.ingredients)
            print("[AI 레시피 추천]")
            for r in recipes:
                print(f"- {r['name']} (예상 선호도: {r['score']:.2f})")

        elif cmd == "6":
            # 사용자가 방금 해 먹은 레시피에 대한 피드백 추가
            ing_text = input("사용한 재료들 (공백으로 나열, 예: 닭고기 양파 마늘): ")
            liked_str = input("이 레시피가 마음에 들었나요? (y/n): ").strip().lower()
            liked = (liked_str == "y")
            ai.add_feedback(ing_text, liked)

        elif cmd == "7":
            # 지금까지 쌓인 피드백으로 AI 모델 재학습
            ai.train_model()

        elif cmd == "8":
            # [NEW] 이미 유통기한이 지난 식재료 보기
            fridge.print_expired_alert()

        elif cmd == "9":
            # [NEW] 식재료 삭제
            fridge.list_ingredients()
            if not fridge.ingredients:
                continue
            idx_str = input("삭제할 식재료 번호 (취소: 엔터): ").strip()
            if not idx_str:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                print("숫자를 입력하세요.")
                continue
            fridge.remove_ingredient(idx)

        elif cmd == "0":
            # 루프 종료 → 프로그램 종료
            print("종료합니다.")
            break

        else:
            print("잘못된 입력입니다. 다시 선택해주세요.")

# 파이썬 파일을 직접 실행했을 때만 main_cli()가 동작하도록 하는 관행적인 코드
if __name__ == "__main__":
    main_cli()
