"""
================================================================================
Copyvisor Description AI - Backend (main.py)
================================================================================
[프로젝트 개요]
- 메타(Meta) 피드 광고용 디스크립션을 자동 생성하는 챗봇형 SaaS 백엔드
- 유저가 이미지를 드래그앤드롭하면 별도의 '생성' 버튼 클릭 없이 즉시
  Gemini 멀티모달 분석 -> 브랜드별 '언어적 DNA' 디스크립션 3종 생성까지 한 번에 처리

[v2 업데이트 사항]
- 5개 브랜드(호갱노노 / 직방 / 네이버페이 / 헤이딜러 / LX하우시스)의
  형태소 패턴, 문장 구조, 이모지 밀도, 글자수 제한, 금지어 규칙을
  BRAND_LINGUISTIC_SPEC에 전체 반영
- Gemini 응답 스키마를 image_analysis / brand_config / generated_copies(ver_1~3)
  구조로 변경 (기존 inferred_persona/copy_variations 구조 폐기)
- 브랜드 간 언어 자산 교차 오염 방지를 위해 프롬프트에 해당 브랜드 스펙만 주입

[기술 스펙]
- FastAPI + google-genai(최신 정식 SDK) + gemini-2.5-flash
- response_mime_type="application/json" 강제 설정으로 구조화된 출력 보장
- 마크다운 코드블록(```json ... ```) 오염 대비 방어적 파싱(clean_json_string) 포함

[실행 전 준비물]
1. pip install fastapi uvicorn python-multipart google-genai pydantic
2. 환경변수 설정: export GEMINI_API_KEY="여기에_본인_API_KEY"
3. 실행: uvicorn main:app --reload --host 0.0.0.0 --port 8000

[추후 확장 포인트]
- 브랜드 스펙은 현재 BRAND_LINGUISTIC_SPEC 딕셔너리에 하드코딩되어 있습니다.
  추후 DB(예: PostgreSQL)에서 브랜드 ID로 조회하는 구조로 교체하시면 됩니다.
- 라우터 분리 시: routers/chat.py 로 generate_copy 엔드포인트를 이동하고
  main.py에는 app.include_router(...)만 남기면 됩니다.
- 응답 JSON의 필드명(generated_copies, ver_1~3, original_copy 등)은 프론트엔드
  index.html과 맞물린 API 계약이므로, 화면 표기를 "디스크립션"으로 바꾸더라도
  필드명 자체는 임의로 바꾸지 않습니다 (바꾸면 프론트 파싱이 깨집니다).
================================================================================
"""

import os
import time
import json
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ------------------------------------------------------------------------------
# 0. 로깅 & 환경설정
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("copyvisor")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
logger.info(f"[INFO] GEMINI_API_KEY 로드 완료: {'설정됨' if GEMINI_API_KEY else '미설정'} (길이: {len(GEMINI_API_KEY)})")

# google-genai 공식 클라이언트 초기화
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("[INFO] Gemini 클라이언트 초기화 성공")
except Exception as e:
    logger.error(f"[ERROR] Gemini 클라이언트 초기화 실패: {e}")
    client = None

GEMINI_MODEL_NAME = "gemini-2.5-flash"


# ------------------------------------------------------------------------------
# 1. FastAPI 앱 초기화
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Copyvisor Description AI Backend",
    description="메타 피드 광고 디스크립션 자동 생성 챗봇 백엔드 (Multi-Brand Linguistic Engine)",
    version="0.2.0",
)

# 프론트(챗봇 UI)와의 통신을 위한 CORS 설정
# TODO: 운영 배포 시 allow_origins를 실제 프론트엔드 도메인으로 제한할 것
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# 2. Multi-Brand Linguistic Specification (브랜드별 언어적 DNA 규칙)
# ------------------------------------------------------------------------------
# 챗봇 1단계 '광고주 브랜드' 퀵버튼 클릭 시 전달되는 brand_id를 키로 사용.
# 각 value의 "raw_spec" 문자열이 곧 Gemini 프롬프트에 그대로 주입되는
# 브랜드 고유 언어 규칙 원문입니다. (사용자가 제공한 스펙 문서를 그대로 보존)
# 추후 DB 연동 시 get_brand_spec() 내부 로직만 교체하면 됩니다.
BRAND_LINGUISTIC_SPEC = {
    "hogangnono": {
        "name": "호갱노노 (Hogangnono)",
        "raw_spec": """
- 페르소나: 정보 비대칭을 찔러주는 은밀하고 집요한 동네 꿀팁 전문가
- 심리 트리거: 손실 회피(Loss Aversion), FOMO(소외 불안), 비교 심리 자극, 호기심, 실용성(목돈 아끼기)
- 텍스트 길이: 공백 포함 80자 ~ 130자 사이 (줄바꿈 가능, 최대 3문장). 모바일 가독성을 위해 1~3줄 이내로 짧고 강렬하게.
- 문법 및 형태소 규칙:
  - 1문장은 반드시 종결 어미가 의문형(`~사람?`, `~을까?`, `~지?`, `~니?`, `~요?`)이거나 친근한 대화체여야 함.
  - 최종 문장은 반드시 명사형 종결이나 단정적 구어체(`~임`, `~하면 됨`, `~중`)로 끝내거나, `~보세요`처럼 행동을 권유하는 구어체로 끝낼 것.
  - '~하십시오' 같은 딱딱한 격식체는 금지하되, '~보세요', '~확인하세요'처럼 친근한 권유형 어미는 허용.
- 이모지 배치 스펙: 의문문 뒤 🤔 또는 👀, 손해/긴급 키워드 뒤 🥲 또는 💥🔥, 비밀/은밀한 정보 뒤 🤫, 강조 시 ‼️✅✔️, 장소/건물 관련 🏢🏠🏘️🚉🏫, 사람 관련 👩‍👧💕. 문장당 1~2개 제한.
- 치트키 키워드: 급매, 시세보다 싼, 호갱, 부동산에선 절대 못 들은 이야기, 거래 급증, 실거래가, 호재, 학군, 역세권, 꿀팁, 전세/매매, 갭투자, 실거주 후기, 청약 시뮬레이션

# 기존 호갱노노 디스크립션 레퍼런스 (이 30개의 톤앤매너, 리듬, 길이감을 반드시 따를 것 — 똑같이 베끼지 말고 동일한 스타일로 새로 쓸 것)
1. 가점 넣어보면 예상 결과까지! 청약 시뮬레이션으로 미리 확인하세요 🏢
2. 슬슬 전세 만기 다가오는 사람? 🤔
3. 분명 같은 단지인데 왜 우리 아파트만 안 오를까? 🥲
4. 올여름 호갱 안 당하려면 이것만 보면 됨 ✅
5. 이 동네 집값 오른다는 거 다들 알고 있었어 🤫
6. 💰 3천만원으로 급매 아파트 잡을 수 있어요. 학군 좋고 역세권까지, 호갱노노에서 지금 바로 확인해보세요
7. 나에게 딱 맞는 집을 찾고 싶다면? 호갱노노 🏠
8. 🏠 같은 가격, 10평 차이. 비교해봐야 아는 정보, 호갱노노에서 지금 확인하세요
9. 🏘️ 지역별로 가격 떨어진 매물만 골라 모았어요. 지금 확인하면 목돈 아낄 수 있어요
10. 시세보다 싸게 살 수 있는 급매 아파트는 어디서 찾지? 👀
11. 🚉 역세권 5분거리에 가격까지 합리적인 아파트, 진짜 있을까요? 🤔
12. 부동산에선 절대 못 들은 이야기 ‼️
13. 🏢 모든 아파트 부동산 정보는 호갱노노에서! 실거래가부터 거래 급증 지역, 개발호재 지역까지
14. 호갱노노에서 남들보다 빠르게 부동산 정보 받아가세요
15. 앱 5개 볼 필요 없습니다. 여기 다 있어요 ✅
16. 몰랐으면 손해 볼 뻔한 단지 정보 💥
17. 👩‍👧 학군 따지는 엄마들이 직접 쓴 동네 후기, 부동산에선 절대 못 들을 이야기
18. 🤔 같은 가격인데 왜 만족도가 다를까? 집값만 보면 절대 모르는 진짜 차이, 지금 확인
19. 💕 신혼집 찾는다면? 이번 주 가장 많이 본 아파트부터 확인해보세요
20. 🤫 우리 동네 실거래 이야기부터 거래량 변화까지, 아는 사람들은 이미 확인 중
21. 📚 진짜 동네 후기부터 시세보다 저렴하게 사는 꿀팁까지 확인
22. 같은 동네도 가격은 다릅니다 ‼️
23. 🔥올여름🔥 뜨는 아파트 모아보기
24. 🏢 1년 만에 6억 오른 아파트, 이유가 궁금하다면?
25. 아파트 정보, 검색보다 호갱노노
26. 시세보다 싼 집, 호갱노노에서 확인 ✔️
27. 🏫 좋은 학군 찾기, 여기서 시작
28. 지금 많이 보는 인기 매물 🏢
29. 장마에도 뽀송한 집이 있다? 쾌적한 아파트 추천 ✔️
30. 👀 올여름 뜨는 아파트 트렌드와 입주 정보까지 한 번에 확인하세요 나랑 제일 맞는 아파트는 호갱노노에서 추천해드려요 🔍

# 디스크립션 작성 시 반드시 아래 5가지 소구 테마 중 이미지 분석 결과와 가장 잘 맞는 테마를 골라 활용할 것
- 테마 1 [불안/의문 자극]: 왜 우리 집만 안 오를까? 같은 손실 회피·비교 심리
- 테마 2 [가성비/급매/자금 맞춤]: 소액 갭투자, 시세보다 싼 집, 목돈 절약
- 테마 3 [실거주자 리얼 후기]: 층간소음, 주차, 학군 등 진짜 주민만 아는 정보
- 테마 4 [트렌드/인기 매물]: 이번 주 가장 많이 본 아파트, 지금 뜨는 지역
- 테마 5 [초보자/편의성 기능]: 청약 시뮬레이션, 앱 하나로 끝내기
""".strip(),
    },
    "zigbang": {
        "name": "직방 (Zigbang)",
        "raw_spec": """
- 페르소나: 시간 낭비와 리스크를 완벽히 제거해 주는 효율 중심의 테크 가이드
- 심리 트리거: 안전성(Security), 신속성(Speed), 실패 없는 선택, 보증금 안심
- 텍스트 길이: 공백 포함 90자 ~ 140자 사이 (단락 구분을 위해 반드시 2회 이상의 줄바꿈(\\n) 적용). 모바일 배너 및 푸시 알림에 적합하도록 단도직입적인 1~3줄 이내로 구성.
- 문법 및 형태소 규칙:
  - 청유형 또는 목적 지향적 명령형 어미(`~확인하세요`, `~찾아보세요`, `~해결💡`)를 디폴트로 사용함.
  - 문장 내에 구체적인 조건절(`~라면`, `~하기 좋은`)을 선행 배치할 것.
  - 깔끔하고 신뢰감을 주는 문체를 유지할 것.
- 이모지 배치 스펙: 문장 시작 또는 끝에 직방 시그니처인 주황색 하트(🧡) 1회 이상 필수 포함. 키워드에 따라 🏠, 🚇, 🌳, 🚨, 🔍, 👀, 🐾, ☀️, ❄️, 📊, 📚, 🏢 매칭.
- 치트키 키워드: 한 번에, 빠르게, 한곳에서, 지킴진단 리포트, 허위 매물, 실거래가 비교, 청약, 분양탭, 전세사기, 깡통전세, 학세권, 역세권, 신축

# 타겟 확장 (직방만의 차별점)
직방은 아파트뿐 아니라 원룸, 오피스텔, 빌라를 구하는 1인 가구, 직장인, 사회초년생까지 폭넓게 다룬다.
이미지 분석 결과 등장하는 인물/공간이 신혼부부, 1인 가구, 직장인, 반려동물 보호자 등 어떤 페르소나에 가까운지 파악하여
출퇴근 동선, 역세권, 반려동물 동반, 숲세권/산책 환경 같은 라이프스타일 요소를 적극 반영할 것.

# 직방만의 핵심 솔루션 (가능하면 자연스럽게 녹여낼 것)
- 허위매물 없는 안전한 매물 검증
- 보증금 안심: 지킴진단 리포트로 깡통전세·전세사기 사전 점검
- 청약부터 신청까지 분양탭에서 원스톱 해결

# 기존 직방 디스크립션 레퍼런스 (이 30개의 톤앤매너, 리듬, 길이감을 반드시 따를 것 — 똑같이 베끼지 말고 동일한 스타일로 새로 쓸 것)
1. 👉원룸·오피스텔·빌라에 대한 모든 부동산 정보는 직방으로 한 번에 확인하세요! 🧡
2. 실거래부터 청약 일정까지 놓치지 말고 직방에서 확인하세요 🏠
3. 거품 없는 시세와 입주민 후기까지 직방에서 비교해보세요 📊
4. 직방에서 조건 맞는 신축 아파트를 빠르게 찾아보세요 🏠
5. 허위 매물은 줄이고, 내 조건 맞는 집만 빠르게 찾아보세요 🏠
6. 가격 메리트 있는 매물만 모아 빠르게 확인하세요 🔍
7. 실거래가 비교로 숨은 알짜 매물만 찾아보세요 🔍
8. 실거래 업데이트부터 청약 정보까지 최신 데이터를 확인하세요 🔍
9. 실거래·청약·입주민 후기까지 필요한 정보를 한곳에서 확인하세요 ✨
10. 첫 신혼집으로 인기 많은 단지와 실거래 정보를 한 번에 확인하세요 🏠
11. 학교 가까운 학세권 아파트와 입주민 후기까지 직방에서 확인하세요 📚
12. 청약 수요가 몰리는 단지부터 한눈에 확인하세요 🏢
13. 산책하기 좋은 단지와 신축 매물 확인하기 🌳
14. 최근 가격 조정된 신축 아파트를 직방에서 빠르게 찾아보세요 👀
15. 가격 비교 안 하면 놓칠 수 있는 숨은 매물, 직방에서 찾아보세요 👀
16. 🔥 갑자기 거래가 몰린 이유 직방 데이터로 먼저 확인하세요 🔍
17. 채광·입지·가격까지 만족도 높은 인기 아파트 👀 지금 확인해보세요.
18. 반려동물 키우기 좋은 집 🐾🐶🏡
19. 여름 되기 전 냉방 효율 좋은 곳으로 ❄️
20. 마트·생활권 가까운 집, 시세 좋은 매물까지 직방에서 확인하세요 👀
21. 위치 좋고, 시세 괜찮고, 채광까지 좋은 집 ☀️
22. 같은 가격이라면 역 가까운 평지 아파트로 🚇 직방에서 추천받아보세요!
23. 야근보다 아까운 건 출퇴근 시간 😮 지금 집값으로 직방에서 더 좋은 아파트 찾아보기 🌳
24. 올여름 이사 갈 집, 아직 못 찾았다면 ⏰🏠
25. 청약 매물 이제 힘들게 찾지 마세요! '직방 분양탭'에서 단지를 모아보고, 신청까지 한 번에 해결💡 청약 신청도, 이제 직방에서 하세요 🧡
26. "지금 가장 빠르게 오르는 경기 핵심" 🏠 우리 동네 아파트 매물 한눈에 👀 직방
27. 🚨직방 [지킴진단 리포트] OPEN 🚨 소중한 내 보증금, 문제 없을지 진단하고 싶다면? 지금, 직방에서 무료로 진단해보세요 🧡
28. 🚨직방 [지킴진단 리포트] OPEN 🚨 알아보고 계신 전셋집, 깡통전세/전세사기로부터 안전하신가요?! 지금, 직방에서 무료로 점검하고 걱정 덜으세요 🧡
29. 3호선 직장인 출퇴근 편한 방! 직방에서 찾기 🚇
30. 시세보다 저렴하게 사는 꿀팁까지 확인. 같은 동네도 가격은 다릅니다 ‼️

# 디스크립션 작성 시 반드시 아래 5가지 소구 테마 중 이미지 분석 결과와 가장 잘 맞는 테마를 골라 활용할 것
- 테마 1 [직장인/라이프스타일 맞춤]: 출퇴근 단축, 지하철 노선별 추천, 반려동물/숲세권
- 테마 2 [보증금 안전/안심 전세]: 지킴진단 리포트, 깡통전세 예방, 전세사기 안심
- 테마 3 [원룸·오피스텔·빌라 탐색]: 가성비 좋은 첫 독립, 허위매물 없는 방 찾기
- 테마 4 [원스톱 청약/분양]: 직방 분양탭, 조건별 청약 일정, 신청까지 한 번에
- 테마 5 [실속 매물/조건 검색]: 가격 조정된 신축, 채광/시세 만족도 높은 집
""".strip(),
    },
    "npay": {
        "name": "네이버페이 (Naver Pay)",
        "raw_spec": """
- 브랜드 정체성: 단순 결제 앱이 아니라 '포인트 적립 + 금융 비교/신청'까지 아우르는 종합 금융 플랫폼.
  타겟이 명확히 두 갈래로 나뉘므로, 이미지에서 감지되는 인물/맥락에 따라 아래 [트랙 A] 또는 [트랙 B] 중
  더 적합한 쪽 하나를 선택하여 그 트랙의 규칙만 적용할 것. 두 트랙의 어휘를 섞어 쓰지 말 것
  (예: 트랙 A의 유행어를 트랙 B의 대출 카피에 쓰지 않는다).

---

## [트랙 A] 10대 Z세대 타겟 — 머니카드Y / 포인트 / 페이펫
- 적용 조건: 이미지에 10대~20대 초반으로 보이는 인물, 학생 분위기, 웹툰/캐릭터/카드 실물, 또래 집단이 등장할 때
- 페르소나: 유행에 민감하고 즉각적 보상에 반응하는 영크크(Young & Trendy) Z세대 친구
- 심리 트리거: 또래 압박(Peer Pressure), 즉각적 리워드(Instant Reward), FOMO
- 텍스트 길이: 공백 포함 60자 ~ 90자 이내 (줄바꿈 절대 금지, 완벽한 단일행 1줄 출력). 단, 친구 초대형/이벤트형처럼 정보량이 많을 경우 최대 130자까지 허용.
- 문법 및 형태소 규칙:
  - 10대 유행어 및 은어(`~특`, `~고고`, `~구움`, `~됐어요`)를 문장 내 자연스럽게 조화시킬 것.
  - 타겟을 직접 도발하거나 공감대를 자극하는 이인칭/반말 어조(`너만 안 받는 거`, `자기 전에 생각날 거야`) 활용.
  - 양적 수치(`최대 1만원`, `쿠키 40개`, `5,000원`)를 구체적으로 노출할 것.
- 이모지 배치 스펙: 문장 내부와 끝에 화려하고 자극적인 이모지(✨, 🍪, ✌️, 💸, 🎁, 💚, 🔥, 👀, 💬, 📢)를 최소 2~3개 이상 밀도 높게 배치.
- 치트키 키워드: 우리 반 친구들, 머니카드 Y, 공짜 웹툰 쿠키, 페이펫, 친구 초대, 포인트, 첫 결제, 삼성페이 연동

### 트랙 A 레퍼런스 카피 (톤앤매너, 리듬을 따를 것 — 똑같이 베끼지 말 것)
1. 우리 반 친구들 다 이미 발급했대요! Npay 카드 만들고 최대 1만원 행운 잡으세요! ✨
2. ✨10대를 위한 첫 카드! 용돈처럼 포인트 쌓이는 나만의 첫 MOVE! 지금 바로 시작하세요. 🚀
3. 내 당첨 포인트는 얼마일까? 💸 10대 전용 머니카드 Y 만들고 포인트 받자! 최대 1만원 포인트 받기 🎁
4. 💬 친구 초대할 때마다 최대 1만원!! 💰 👉 친구 초대하고 포인트 혜택 받기
5. 자기 전에 생각날 거야..Npay 머니카드 Y 진짜 많이 도움된다 💪 용돈처럼 쌓이는 포인트로 굿 소비하자
6. 👀보고싶은 작품 쌓여있다고?! 쿠키 40개 받고 웹툰 공짜로 보자 👉Npay 머니카드 Y 혜택 받고 정주행 고고 🛻
7. 📢 여러분 저 됐어요.. 포인트 부자 됐어요!! Npay 머니카드 Y 발급 시 최대 1만원 💚 편의점, 다이소 등 현장 결제하면 포인트 두 번 뽑기까지 ✌️
8. 요즘 10대 특) 웹툰 쿠키 공짜로 구움 🍪 10대가 웹툰 쿠키를 왜이리 잘구워..✨ 공짜 쿠키 40개 주는 Npay 머니카드Y
9. 쓰던 삼성페이 방식 그대로인데, 혜택은 무려 두 배! ✌️ 기존 카드 혜택에, 네이버페이 포인트 추가 적립까지! 💰 지금 신청하고 첫 결제하면 5,000원 모두 드려요! 🔥
10. 우리 반 친구들 다 이미 발급했대요! 네이버페이 머니카드 Y 만들고 최대 1만원 뽑기✨ 💌친구 초대 할 때 마다, 포인트 뽑기 한 번 더!
11. 이미 쓰고 있는 삼성페이, 네이버페이 포인트도 함께 쌓으세요. 🔥 첫 결제 금액의 100%, 5,000원이 그대로 내 포인트로!
12. ⚽ 내가 바로 축구왕..?! 축구왕 페이펫 출격! 🎁

---

## [트랙 B] 금융/대출 수요자 타겟 — 햇살론 / 대출비교 / 저신용자
- 적용 조건: 이미지에 성인, 직장인, 가계/금융 관련 소품(통장, 계산기, 서류, 한숨 쉬는 표정 등)이 등장하거나
  대출/자금 관련 맥락이 감지될 때
- 페르소나: 신용점수나 소득에 위축된 사람도 안심시켜주는 차분하고 신뢰감 있는 금융 가이드
- 심리 트리거: 안심(저신용자도 가능), 비교를 통한 합리적 선택, 긴급 자금 니즈 해소
- 텍스트 길이: 공백 포함 40자 ~ 90자 이내. 정보 전달이 핵심이므로 과도한 줄바꿈 없이 1~2문장으로 간결하게.
- 문법 및 형태소 규칙:
  - 정중하지만 부담스럽지 않은 권유형 어미(`~보세요`, `~가능!`, `~확인해보세요`)를 사용할 것.
  - 자격 조건이나 한계를 먼저 짚어주고(`신용점수, 소득 낮아도`, `대출 많아도`) 그럼에도 가능하다는 구조로 안심시킬 것.
  - 트랙 A의 유행어, 반말, 과도한 이모지는 절대 사용하지 말 것 — 신뢰감이 최우선.
- 이모지 배치 스펙: 문장당 0~1개로 절제. 사용한다면 💸 정도만 허용. 화려한 이모지(✨🎁💚 등) 금지.
- 치트키 키워드: 햇살론, DSR, 저신용자, 신용점수, 대출비교, 한도조회, 당일 입금, 1금융권

### 트랙 B 레퍼런스 카피 (톤앤매너를 따를 것 — 똑같이 베끼지 말 것)
1. DSR 부담 덜어주는 햇살론! 네이버페이에서 한도조회부터 신청까지 한번에 가능!
2. 신용점수, 소득 낮아도 금리 5%대 대출신청 가능! 네이버페이로 햇살론 대출 비교해보세요
3. 오늘 당장 비상금 필요하다면? 당일 입금 가능한 대출찾기도 네이버페이에서!
4. 햇살론이라면 저신용자도 1금융권 대출신청 가능! 네이버페이에서 확인해보세요
5. 이미 대출 많아도, 네이버페이에서 한번 더 대출비교 해보세요!
6. 600점대 저신용자도 대출 신청 성공! Npay에서 가능 상품 확인해보세요 💸
""".strip(),
    },
    "heydealer": {
        "name": "헤이딜러 (Heydealer)",
        "raw_spec": """
- 페르소나: 중고차 시장의 불신과 조작을 심판하는 냉철하고 단호한 블랙 페르소나. 가볍게 붕 뜨는 유행어 대신, 날카롭고 신뢰감을 주며 브랜드 철학이 느껴지는 묵직한 문장을 구사함.
- 심리 트리거: 인지 부조화(Cognitive Dissonance), 원천적 불신 제거, 극단적 파격, 압도적 가격 혜택(100만원 이벤트)
- 텍스트 길이: 공백 포함 15자 ~ 50자 이내. 사족을 제거한 짧고 단호한 1줄, 혹은 짧은 두 문장 조합.
- 문법 및 형태소 규칙:
  - 주어+목적어+서술어가 극도로 압축된 형태를 취함. 조사가 생략된 명사형 종결(`~한 존재`, `~한 차만 판매`, `~하는 법`)이나 단호한 평서형 종결(`~드려요`, `~판매합니다`, `~확인해보세요`)을 사용.
  - 감탄사나 과장된 수식어(진짜, 완전, 대박, 추천 등) 사용 시 즉시 탈락 처리. 감정적 어조 0% 유지 — 담담하고 단호하게 사실만 전달.
  - [곧마감]처럼 대괄호로 긴급성을 압축해 표현하는 방식을 적극 활용할 것.
- 이모지 배치 스펙: 자동차 관련(🚗, 🚘) 또는 경고/강조(🚨, 📢) 이모지를 문장당 0~1개, 꼭 필요한 곳에만 정제되어 사용. 화려하거나 감정적인 이모지(✨💕🎉 등)는 절대 금지 — 전문적인 느낌을 해치지 않는 선에서만 허용.
- 치트키 키워드: 100만 원, 인간의 의심, 무사고, 하부누유 없음, 조작 불가, 360도, 곧마감, 한정 세일

# 기존 헤이딜러 디스크립션 레퍼런스 (이 톤앤매너, 압축된 리듬을 반드시 따를 것 — 똑같이 베끼지 말고 동일한 스타일로 새로 쓸 것)
1. 새로운 눈이 나타났다.
2. [곧마감] 모든 차, 100만원에 드려요
3. 인간의 의심이 만든 세상에 없던 존재
4. 이번 달 한정 세일 차량 확인해보세요
5. 중고차 실내, 외관 360도로 확인
6. 100% 무사고차량만 판매
7. 1억 넘는 포르쉐 100만원에 사는 법
8. 하부누유 없는 중고차만 판매합니다.
9. GV70, 100만원에 드려요

# 디스크립션 작성 시 반드시 아래 5가지 소구 테마 중 이미지 분석 결과와 가장 잘 맞는 테마를 골라 활용할 것
- 테마 1 [파격적 100만원 이벤트/래플]: 수입차/인기 SUV를 100만 원에 잡을 기회, 곧 마감 같은 압도적 가격 후킹
- 테마 2 [의심을 지우는 철저한 진단]: 하부 누유 zero, 침수차 프리패스 방지, 360도 정밀 확인
- 테마 3 [브랜드 철학 및 감각적 메시지]: 중고차 시장의 새로운 기준, 인간의 의심이 만든 눈 같은 철학적·날카로운 한 줄
- 테마 4 [무사고 / 한정 세일 혜택]: 100% 무사고 인증 차량, 이번 달만 가능한 한정 특가 매물
- 테마 5 [타겟 저격 내 차 사기/팔기 편리성]: 직원이 직접 진단하는 편리함, 감가 걱정 없는 거래
""".strip(),
    },
    "lx_hausys": {
        "name": "LX하우시스 (LX Hausis)",
        "raw_spec": """
- 페르소나: 주거 공간의 가치와 미학을 제안하는 프리미엄 인테리어 트렌드 큐레이터
- 심리 트리거: 심미적 만족감(Aesthetic), 하이엔드 라이프스타일, 시공에 대한 절대적 신뢰, 계절 변화에 대비하는 실질적 안심감
- 텍스트 길이: 공백 포함 250자 ~ 450자 이내 (줄바꿈과 문단 구분을 통해 가독성 있는 정제된 롱폼 콘텐츠 구조). 문단 사이에는 적절한 여백을 두어 매거진 칼럼처럼 호흡을 줄 것.
- 문법 및 형태소 규칙:
  - 인테리어 잡지나 매거진 에세이 톤의 우아하고 격식 있는 평서문 어미(`~보세요`, `~완성합니다`, `~더합니다`, `~확인해 보세요`)를 구사할 것. 가벼운 유행어는 절대 지양.
  - 단순 기능 나열을 배제하고 감각적인 형용사(`감도 높은`, `깊이감 있는`, `입체적인 결`, `우아한 색 조합`, `묵직한`, `차분한 인상`)를 풍부하게 활용.
  - 단순히 "예쁜 벽지", "좋은 바닥재"라고 뭉뚱그리지 말고, 빛의 각도에 따른 음영, 마블 패턴의 미세한 컬러 차이, 질감의 두께처럼 시각적·촉각적 디테일을 구체적으로 묘사할 것.
  - 실제 제품군 카테고리와 구체적인 컬러칩/제품 라인업 이름, 아파트 평형수 정보(`32평`, `35평`, `디아망 회벽`, `하이막스 오로라 블랑`, `강마루 프리미엄 합판`)를 본문 중간에 명확히 임베딩할 것.
  - 계절 변화(장마철 습기, 곰팡이, 결로, 겨울철 단열 등)에 따른 기능성·내구성을 짚어주며 실질적인 솔루션으로 안심시킬 것.
- 이모지 배치 스펙: 문단의 첫 시작이나 강조 지점에 제목용 기호(💡, ✔, ✓, 🌟, 🎁)를 절제되고 고급스럽게 활용. 체크표시(✔)로 핵심 장점을 리스트업하는 구성도 적극 활용. 감정 자극형 이모지(✨😍🔥 등)는 배제.
- 치트키 키워드: 감도 높은, 깊이감, 오브제, 책임 시공, Before & After, 이달의 무드, 플래그십, 디자인 키트, 큐레이션, 시공 사례

# 기존 LX하우시스 디스크립션 레퍼런스 (이 9개의 매거진 에세이 톤앤매너, 호흡, 디테일 묘사 방식을 반드시 따를 것 — 똑같이 베끼지 말고 동일한 스타일로 새로 쓸 것)
1. 어떤 자재 조합이 우리 집에 어울릴지 그려지지 않을 때, 분위기부터 정하면 방향이 보입니다. 공간을 구성하는 자재를 스타일별로 큐레이션한 LX지인의 '디자인 키트', 이달의 무드는 #웜베이지 입니다. 크림 톤과 내추럴 오크의 조화로 아늑하고 편안하게 완성한 공간의 분위기를 영상으로 직접 확인해 보세요. 원하는 무드를 고르는 것만으로 공간의 방향이 잡힙니다. 마음에 드는 자재의 조합을 직접 눈으로 확인하고 싶다면 LX지인 플래그십을 방문해 보세요.
2. LX지인 인조대리석 하이막스 오로라&칼라카타 오로라 블랑 VS 오로라 웜블랑. 어떤 컬러가 우리 집 주방에 맞을까요? 비슷해 보이는 아이보리 톤이지만 컬러감과 마블 패턴에서 분명한 차이가 있습니다. 화이트에 가까운 맑은 오로라 블랑부터 따뜻한 웜 아이보리 톤에 그레이 마블 패턴이 더해진 오로라 웜블랑까지, 우리 집 주방에 맞는 하이막스를 선택해 보세요.
3. 높은 습도와 잦은 비로 누수와 곰팡이 우려가 커지는 장마철. 막연한 걱정 대신 '사전 점검 - 예방 - 대처' 3단계만 미리 알아두면 생활 속 대부분의 피해를 줄일 수 있습니다. 장마 전 창호·외벽·배수구 점검 방법부터 결로와 습기가 몰리기 쉬운 창호 주변 관리법까지. 장마철 우리 집을 지키는 곰팡이·누수 방지 가이드를 확인해 보세요.
4. LX지인 아카이브 | 디아망 회벽. 매끄럽게 칠한 페인트도, 패턴을 찍어낸 벽지도 아닙니다. 미장한 듯 겹겹이 쌓아 올린 질감의 벽지, #디아망 회벽을 소개합니다. 도톰한 두께 위에 구현한 섬세한 회벽 질감은 빛이 닿는 각도에 따라 미묘한 음영을 만들며 공간에 색다른 깊이감을 더합니다. 매끈한 벽에서는 느낄 수 없던 입체적인 결이 가장 넓은 면을 하나의 오브제로 완성합니다. 벽에서 시작되는 공간의 변화, LX지인 벽지 디아망 회벽으로 완성해 보세요.
5. 창호 교체 전 브랜드 비교는 해보셨나요? 제품, 가격, 시공, A/S 무엇이 다른지 확인해 보세요! 우수한 단열 성능과 디자인으로 꾸준한 사랑을 받고 있는 LX Z:IN 창호, 💡LX Z:IN 창호, 뭐가 다를까요? ✔다양한 수상으로 입증한 제품력 ✔경쟁력 있는 가격과 다양한 혜택 ✔책임 시공 및 책임 A/S 제품부터 시공, 사후 관리까지 완벽하게! 믿을 수 있는 LX Z:IN으로 우리집 창호를 바꿔보세요🌟
6. 32평 아파트, 우아하고 쾌적하게 바꾸고 싶다면? 벽지와 바닥재는 아파트 인테리어의 분위기와 개방감을 좌우하는 중요한 요소입니다. 깊이 있는 회벽 질감이 돋보이는 LX지인 벽지 디아망 회벽/블랑 그레이를 중심으로 완성한 32평 아파트 시공사례를 소개합니다. 우아한 색 조합과 실용적인 설계가 돋보이는 32평 아파트 인테리어를 직접 확인해 보세요.
7. 밝고 깨끗한 화이트 우드 주방에 감각적인 컬러 포인트를 더했습니다. 벽지 #베스트 와 바닥재 #강마루 로 내추럴한 분위기를 살리고, 주방에는 키친 #셀렉션5 와 인조대리석 #하이막스 를 적용해 화이트 우드 주방의 깔끔한 바탕을 완성했습니다. 여기에 그린 컬러 타일을 매치해 산뜻한 생기를 더했습니다. 그린 컬러 포인트로 완성한 35평 아파트, Before & After 를 확인해 보세요.
8. 습도 관리가 중요한 장마철에는 바닥재의 들뜸과 변형 우려까지 세심하게 살펴보게 됩니다. LX지인 바닥재 강마루 프리미엄 합판(강그린와이드)는 계절 변화가 큰 시기에도 안정적인 바닥 상태를 유지할 수 있도록 내구성을 높인 기능성 바닥재입니다. 묵직한 깊이감이 돋보이는 로얄 월넛, 밝고 차분한 인상의 어니스트 오크, 산뜻하고 내추럴한 라임 오크까지. 시공 사례를 통해 직접 확인해 보세요.
9. 인테리어 상담만 해도 혜택이 트리플로 UP! 🎁 단열 성능 UP, 공간 활용 UP, 주방 품격 UP까지 LX지인이 준비한 최대 216만 원 상당 혜택을 만나보세요. 터닝도어 + 인테리어용 단열재 + 베버트 수전까지 지금 상담 신청하고 특별 혜택 받아보세요!

# 디스크립션 작성 시 반드시 아래 5가지 소구 테마 중 이미지 분석 결과와 가장 잘 맞는 테마를 골라 활용할 것
- 테마 1 [시즌별 홈 케어 & 기능성 솔루션]: 장마철 습기 걱정 없는 강마루, 겨울철 단열을 위한 창호 교체
- 테마 2 [공간 무드 & 자재 큐레이션]: 디자인 키트, 내추럴 우드/모던 그레이 무드 제안
- 테마 3 [자재 디테일 비교 및 소개]: 디아망 벽지의 질감, 하이막스 인조대리석 패턴 차이
- 테마 4 [실제 평형별 시공 사례 / B&A]: 20~30평대 아파트 리모델링, 주방/거실 공간의 변화
- 테마 5 [플래그십 방문 및 상담 이벤트]: 지인 플래그십 스토어 초청, 상담 시 트리플 업 혜택
""".strip(),
    },
}

# 시스템 공통 절대 금지 규칙 (모든 브랜드에 공통 적용, 프롬프트 하단에 항상 주입)
GLOBAL_NEGATIVE_CONSTRAINTS = """
1. 문장 끝에 의미 없는 온점(.)을 남발하지 말 것. (브랜드별 어미 가이드를 최우선 준수)
2. 지정된 브랜드의 길이 제한(글자 수 및 줄바꿈 여부)을 어길 시 에러로 간주함.
3. 브랜드 간의 시그니처 언어 자산(예: LX하우시스에 10대 유행어를 쓰거나, 헤이딜러에 화려하고 감정적인 이모지를 쓰는 등)이 교차 오염되는 것을 엄격히 금지함. 각 브랜드의 raw_spec에 명시된 이모지/어휘 규칙만 따를 것.
""".strip()


def get_brand_spec(brand_id: str) -> dict:
    """
    brand_id로 언어적 DNA 스펙을 조회.
    추후 이 함수 내부만 DB 쿼리 로직으로 교체하면 나머지 코드는 그대로 재사용 가능.
    """
    spec = BRAND_LINGUISTIC_SPEC.get(brand_id)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"등록되지 않은 브랜드입니다: '{brand_id}'. "
                   f"사용 가능한 브랜드: {list(BRAND_LINGUISTIC_SPEC.keys())}",
        )
    return spec


# ------------------------------------------------------------------------------
# 3. Pydantic 스키마 정의
# ------------------------------------------------------------------------------

class ImageAnalysis(BaseModel):
    """Gemini가 이미지에서 스스로 추론한 타겟/무드 분석 결과"""
    detected_target: str = Field(..., description="이미지에서 도출된 핵심 타겟 연령 및 페르소나")
    visual_context: str = Field(..., description="계절감, 지배적 색상, 인물의 상태 분석")


class BrandConfig(BaseModel):
    """적용된 브랜드와 강제 규칙 요약"""
    applied_brand: str = Field(..., description="선택된 브랜드명")
    enforced_rules: str = Field(..., description="글자 수 제한, 줄바꿈 횟수, 이모지 밀도, 금지어 필터링 요약")


class GeneratedCopies(BaseModel):
    """브랜드 스펙을 100% 충족하는 디스크립션 3종"""
    ver_1: str
    ver_2: str
    ver_3: str


class GenerateCopyResponse(BaseModel):
    """
    /api/chat/generate 엔드포인트 최종 응답 스키마.
    프론트(챗봇 UI)는 이 구조 그대로 렌더링하면 됩니다.
    """
    brand_id: str
    image_analysis: ImageAnalysis
    brand_config: BrandConfig
    generated_copies: GeneratedCopies


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class RefineCopyRequest(BaseModel):
    """
    /api/chat/refine 요청 바디.
    이미지 재업로드 없이, 이미 생성된 디스크립션 텍스트 1개를 더 짧게/길게 다시 쓴다.
    """
    brand_id: str = Field(..., description="해당 디스크립션이 속한 브랜드 ID")
    original_copy: str = Field(..., description="더 짧게/길게 조정할 원본 디스크립션 텍스트")
    direction: str = Field(..., description="'shorter'(더 짧게) 또는 'longer'(더 길게)")


class RefineCopyResponse(BaseModel):
    refined_copy: str = Field(..., description="재작성된 디스크립션 텍스트")


# ------------------------------------------------------------------------------
# 4. Gemini 응답 JSON 방어적 파싱 유틸
# ------------------------------------------------------------------------------

def clean_json_string(raw_text: str) -> str:
    """
    Gemini가 response_mime_type="application/json"을 설정해도
    드물게 ```json ... ``` 마크다운 코드블록을 앞뒤로 붙이는 경우가 있어
    이를 안전하게 제거하는 방어 함수.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]

    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]

    return cleaned.strip()


def safe_parse_gemini_json(raw_text: str) -> dict:
    """
    clean_json_string으로 정제 후 json.loads 시도.
    실패 시 명확한 에러 메시지와 함께 HTTPException 발생.
    """
    cleaned_text = clean_json_string(raw_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"[JSON PARSE ERROR] Gemini 원본 응답: {raw_text}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini 응답을 JSON으로 파싱하는 데 실패했습니다. "
                f"파싱 에러: {str(e)}"
            ),
        )


# ------------------------------------------------------------------------------
# 5. Gemini 호출용 프롬프트 빌더 (Multi-Brand Linguistic Specification 적용)
# ------------------------------------------------------------------------------

def build_system_prompt(brand_name: str, brand_raw_spec: str) -> str:
    """
    사용자가 제공한 'Multi-Brand Linguistic Specification' 원문을 그대로
    프롬프트에 임베딩합니다. 단, 선택된 브랜드의 스펙만 주입하여
    브랜드 간 언어 자산 교차 오염을 원천 차단합니다.
    """
    return f"""
# Role
너는 각 브랜드의 미디어 바잉 데이터를 기반으로 CTR(클릭률)을 예측하고, 브랜드 고유의 '언어적 DNA'를 100% 일치시켜 디스크립션을 생산하는 AI 디스크립션 라이팅 엔진이다.
제공된 이미지를 분석한 뒤, 지정된 브랜드의 [형태소 패턴], [문장 구조 규칙], [금지어 가이드라인]을 엄격히 준수하여 디스크립션을 생성하라.

---

# 선택 브랜드: {brand_name}

{brand_raw_spec}

---

# 🚫 시스템 공통 절대 금지 규칙 (Negative Constraints)
{GLOBAL_NEGATIVE_CONSTRAINTS}

---

# Execution & Output Format
입력된 이미지의 [타겟]과 [시각적 무드]를 분석하고, 위 [선택 브랜드]에 지정된 규칙만을 매트릭스로 연산하여 JSON으로 출력하라.
markdown block(```json) 기호는 텍스트 파싱을 방해하므로 절대로 출력하지 말고, 순수 JSON String만 반환하라.

{{
  "image_analysis": {{
    "detected_target": "이미지에서 도출된 핵심 타겟 연령 및 페르소나",
    "visual_context": "계절감, 지배적 색상, 인물의 상태 분석"
  }},
  "brand_config": {{
    "applied_brand": "{brand_name}",
    "enforced_rules": "글자 수 제한, 줄바꿈 횟수, 이모지 밀도, 금지어 필터링 요약"
  }},
  "generated_copies": {{
    "ver_1": "선택 브랜드의 스펙을 100% 충족하는 디스크립션 01",
    "ver_2": "선택 브랜드의 스펙을 100% 충족하는 디스크립션 02",
    "ver_3": "선택 브랜드의 스펙을 100% 충족하는 디스크립션 03"
  }}
}}
""".strip()


# ------------------------------------------------------------------------------
# 6. Gemini 멀티모달 호출 함수
# ------------------------------------------------------------------------------

def call_gemini_for_ad_copy(
    image_bytes: bytes,
    image_mime_type: str,
    brand_name: str,
    brand_raw_spec: str,
) -> dict:
    """
    gemini-2.5-flash를 멀티모달(이미지+텍스트)로 호출하여
    구조화된 JSON 형태의 브랜드 언어적 DNA 디스크립션 세트를 반환받는다.
    """
    system_prompt = build_system_prompt(brand_name, brand_raw_spec)

    try:
        response = gemini_call_with_retry(lambda: client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                # 이미지 파트: 업로드된 바이트를 그대로 inline으로 전달
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
                # 텍스트 파트: 브랜드 언어적 DNA 스펙 + 출력 형식 지시
                system_prompt,
            ],
            config=types.GenerateContentConfig(
                # [필수 스펙] 순수 JSON만 반환받기 위한 설정
                response_mime_type="application/json",
                # 브랜드 어미/형태소 규칙을 엄격히 지키되, 디스크립션 표현의 다양성은 확보
                temperature=0.85,
            ),
        ))
    except HTTPException:
        raise  # gemini_call_with_retry가 변환한 HTTPException은 그대로 올림
    except Exception as e:
        logger.exception("[GEMINI API ERROR]")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}",
        )

    if not response.text:
        raise HTTPException(
            status_code=502,
            detail="Gemini가 빈 응답을 반환했습니다.",
        )

    # [필수 방어 코드] 마크다운 블록 오염 대비 후 JSON 파싱
    parsed_json = safe_parse_gemini_json(response.text)
    return parsed_json


# ------------------------------------------------------------------------------
# 5-1. 글자 수 카운트 유틸 (더 짧게/더 길게 ±20% 검증용)
# ------------------------------------------------------------------------------

def count_chars(text: str) -> int:
    """
    디스크립션 텍스트의 글자 수를 센다 (공백 포함, 원문 그대로).
    '더 짧게/더 길게' 버튼은 이 글자 수를 기준으로 직전 텍스트 대비 ±20%를 목표로 한다.
    """
    return len(text) if text else 0


# ------------------------------------------------------------------------------
# 5-2. Gemini 503 자동 재시도 유틸
# ------------------------------------------------------------------------------

def is_503_error(e: Exception) -> bool:
    """
    Gemini SDK가 던지는 예외 중 503 UNAVAILABLE(일시적 과부하)인지 판별한다.
    - google.genai.errors.ServerError / google.api_core.exceptions.ServiceUnavailable
    - 혹은 에러 메시지에 '503' 또는 'UNAVAILABLE'이 포함된 경우
    """
    msg = str(e).upper()
    return "503" in msg or "UNAVAILABLE" in msg or "HIGH DEMAND" in msg


def gemini_call_with_retry(call_fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Gemini API 호출 함수(call_fn)를 실행하고,
    503 UNAVAILABLE이 발생하면 지수 백오프(2s → 4s → 8s)로 최대 max_retries회 재시도한다.
    다른 에러는 재시도 없이 즉시 re-raise한다.

    사용 예:
        response = gemini_call_with_retry(lambda: client.models.generate_content(...))
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            if is_503_error(e):
                last_exc = e
                wait = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s
                logger.warning(
                    f"[GEMINI 503] attempt={attempt}/{max_retries}, "
                    f"{wait:.0f}초 대기 후 재시도합니다. 에러: {e}"
                )
                time.sleep(wait)
            else:
                raise  # 503 외 에러는 재시도 없이 즉시 올림
    # max_retries번 모두 503이면 마지막 예외를 HTTPException으로 변환
    raise HTTPException(
        status_code=503,
        detail=(
            f"Gemini 서버가 일시적으로 과부하 상태입니다. "
            f"{max_retries}회 재시도 후에도 응답을 받지 못했습니다. "
            f"잠시 후 다시 시도해 주세요. (원인: {last_exc})"
        ),
    )

def build_refine_prompt(brand_name: str, brand_raw_spec: str, original_copy: str, direction: str) -> str:
    """
    이미 생성된 디스크립션 1개를 같은 브랜드 언어 규칙을 유지한 채
    직전 텍스트 기준 글자 수 ±20%로 다시 작성시키는 프롬프트.
    이미지 재분석 없이 텍스트 단독으로 처리한다.
    """
    original_char_count = count_chars(original_copy)
    # 20% 증감분은 최소 1글자는 되도록 보정 (아주 짧은 카피에서 0이 되는 것 방지)
    delta = max(1, round(original_char_count * 0.2))
    target_char_count = (
        max(1, original_char_count - delta) if direction == "shorter"
        else original_char_count + delta
    )
    # Gemini가 정확히 한 글자 단위까지 맞추긴 어려우므로, ±15% 오차까지는 허용 범위로 둔다
    tolerance = max(1, round(target_char_count * 0.15))
    target_min = max(1, target_char_count - tolerance)
    target_max = target_char_count + tolerance

    direction_instruction = {
        "shorter": (
            f"[원본 디스크립션]은 공백 포함 {original_char_count}자다. "
            f"결과는 원본보다 약 20% 짧은, 공백 포함 {target_char_count}자 안팎(허용 범위 {target_min}~{target_max}자)으로 만들어라. "
            "문장이나 절을 통째로 들어내거나, 수식어·부연 표현을 덜어내는 방식으로 분량을 줄여라. "
            "핵심 메시지(브랜드/제품/혜택)는 반드시 살리고, 핵심과 무관한 표현부터 우선 제거하라."
        ),
        "longer": (
            f"[원본 디스크립션]은 공백 포함 {original_char_count}자다. "
            f"결과는 원본보다 약 20% 긴, 공백 포함 {target_char_count}자 안팎(허용 범위 {target_min}~{target_max}자)으로 만들어라. "
            "부연 설명, 구체적인 디테일, 또는 추가 혜택/CTA 중 브랜드 맥락에 가장 자연스러운 내용을 덧붙여 분량을 늘려라. "
            "원본의 핵심 메시지와 어순은 가능한 한 유지하면서 자연스럽게 확장하라."
        ),
    }[direction]

    return f"""
# Role
너는 브랜드 고유의 '언어적 DNA'를 100% 일치시켜 디스크립션을 다듬는 AI 디스크립션 에디터다.
아래 [원본 디스크립션]을 [선택 브랜드]의 언어 규칙을 그대로 유지한 채, [수정 지시]에 따라 다시 작성하라.

---

# 선택 브랜드: {brand_name}

{brand_raw_spec}

---

# 원본 디스크립션 (직전에 생성/조정된 최신 버전)
{original_copy}

# 수정 지시 — 반드시 지킬 것
{direction_instruction}

[우선순위] 위 [선택 브랜드] 규칙에 적힌 글자 수 제한은 "기본 디스크립션을 새로 만들 때"의 가이드다.
지금은 이미 만들어진 디스크립션을 [수정 지시]에 명시된 목표 글자 수로 재조정하는 작업이므로,
이번 한 번에 한해 목표 글자 수({target_min}~{target_max}자)를 브랜드 가이드의 글자 수 범위보다 우선 적용하라.
단, 어미 스타일·이모지 사용 방식·키워드 같은 브랜드의 언어적 DNA 자체는 그대로 유지해야 한다.

---

# 🚫 시스템 공통 절대 금지 규칙 (Negative Constraints)
{GLOBAL_NEGATIVE_CONSTRAINTS}

---

# Execution & Output Format
markdown block(```json) 기호는 절대 출력하지 말고, 순수 JSON String만 반환하라.

{{
  "refined_copy": "수정된 디스크립션 텍스트 1개 (목표 글자 수 범위를 충족할 것)"
}}
""".strip()


def call_gemini_for_refine(brand_name: str, brand_raw_spec: str, original_copy: str, direction: str) -> str:
    """
    이미지 없이 텍스트 전용으로 Gemini를 호출해 디스크립션 1개를
    직전 텍스트(original_copy) 기준 글자 수 ±20%로 재작성한다.

    [핵심 방어 로직] Gemini는 프롬프트로 목표 글자 수를 지시해도
    브랜드 스펙의 글자 수 가이드와 헷갈려 목표를 못 맞추는 경우가 있다.
    이를 막기 위해, 응답을 받을 때마다 실제 글자 수를 코드로 직접 세어
    허용 오차 범위 안에 들어오는지 검증하고, 벗어나면 더 강하게 지시한
    프롬프트로 재시도한다.
    """
    original_char_count = count_chars(original_copy)
    delta = max(1, round(original_char_count * 0.2))
    target_char_count = (
        max(1, original_char_count - delta) if direction == "shorter"
        else original_char_count + delta
    )
    tolerance = max(1, round(target_char_count * 0.15))
    target_min = max(1, target_char_count - tolerance)
    target_max = target_char_count + tolerance

    max_attempts = 3
    last_result = None

    for attempt in range(1, max_attempts + 1):
        # 재시도 시에는 직전 실패 결과를 보여주며 더 단호하게 재지시한다.
        emphasis = ""
        if attempt > 1 and last_result is not None:
            actual_count = count_chars(last_result)
            emphasis = f"""

[재시도 - {attempt}차 시도]
직전 응답: "{last_result}" (공백 포함 {actual_count}자)
이 응답은 목표 글자 수 범위({target_min}~{target_max}자)를 충족하지 못했다. 절대 같은 실수를 반복하지 말고,
이번에는 반드시 공백 포함 {target_min}~{target_max}자 사이로 정확히 출력하라."""

        prompt = build_refine_prompt(brand_name, brand_raw_spec, original_copy, direction) + emphasis

        temperature = max(0.3, 0.7 - (attempt - 1) * 0.2)
        try:
            response = gemini_call_with_retry(lambda: client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # 재시도할수록 온도를 살짝 낮춰 지시 순응도를 높인다.
                    temperature=temperature,
                ),
            ))
        except HTTPException:
            raise  # gemini_call_with_retry가 변환한 HTTPException은 그대로 올림
        except Exception as e:
            logger.exception("[GEMINI API ERROR - refine]")
            raise HTTPException(
                status_code=502,
                detail=f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}",
            )

        if not response.text:
            raise HTTPException(status_code=502, detail="Gemini가 빈 응답을 반환했습니다.")

        parsed_json = safe_parse_gemini_json(response.text)
        refined = parsed_json.get("refined_copy")
        if not refined:
            raise HTTPException(
                status_code=502,
                detail="Gemini 응답에 refined_copy 필드가 없습니다.",
            )

        last_result = refined
        actual_char_count = count_chars(refined)

        logger.info(
            f"[REFINE] attempt={attempt}, direction={direction}, "
            f"original={original_char_count}자, target={target_char_count}자 "
            f"(허용범위 {target_min}~{target_max}자), actual={actual_char_count}자"
        )

        if target_min <= actual_char_count <= target_max:
            return refined

    # max_attempts번 시도해도 허용 범위를 못 맞춘 경우,
    # 완전히 실패 처리하는 대신 마지막으로 받은 결과를 반환한다
    # (사용자 입장에선 약간 안 맞아도 결과가 나오는 게 아예 에러가 나는 것보다 낫다).
    logger.warning(
        f"[REFINE] {max_attempts}회 시도 후에도 목표 글자 수 범위 미달성. "
        f"마지막 결과를 그대로 반환합니다. (target={target_min}~{target_max}자)"
    )
    return last_result


# ------------------------------------------------------------------------------
# 7. 메인 엔드포인트: POST /api/chat/generate
# ------------------------------------------------------------------------------

@app.post(
    "/api/chat/generate",
    response_model=GenerateCopyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청 (이미지 형식 오류 등)"},
        404: {"model": ErrorResponse, "description": "존재하지 않는 브랜드 ID"},
        502: {"model": ErrorResponse, "description": "Gemini API 호출/파싱 오류"},
    },
    summary="이미지 업로드 즉시 브랜드 언어적 DNA 기반 광고 디스크립션 3종 자동 생성",
)
async def generate_copy(
    brand_id: str = Form(
        ...,
        description=(
            "유저가 챗봇 하단 퀵버튼에서 선택한 광고주 브랜드 ID "
            "(예: 'hogangnono', 'zigbang', 'npay', 'heydealer', 'lx_hausys')"
        ),
    ),
    image: UploadFile = File(
        ...,
        description="챗봇 창에 드래그앤드롭으로 업로드된 광고 소재 이미지",
    ),
):
    """
    [UX 시나리오 매핑]
    - 프론트엔드는 유저가 이미지를 드롭하는 즉시(별도 버튼 클릭 없이)
      이 엔드포인트를 자동으로 호출해야 합니다. (multipart/form-data)
    - brand_id는 1단계에서 유저가 클릭한 퀵 버튼 값을 그대로 전달받습니다.
    """

    # --- 7-1. 이미지 파일 유효성 검증 ---
    allowed_mime_types = ("image/jpeg", "image/png", "image/webp")
    if image.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"지원하지 않는 이미지 형식입니다: {image.content_type}. "
                f"지원 형식: {allowed_mime_types}"
            ),
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어있습니다.")

    # --- 7-2. 브랜드 언어적 DNA 스펙 조회 ---
    brand_spec = get_brand_spec(brand_id)

    # --- 7-3. Gemini 멀티모달 호출 (이미지 분석 + 브랜드 언어 DNA 디스크립션 생성을 한 번에) ---
    logger.info(f"[REQUEST] brand_id={brand_id}, image_size={len(image_bytes)} bytes")

    gemini_result = call_gemini_for_ad_copy(
        image_bytes=image_bytes,
        image_mime_type=image.content_type,
        brand_name=brand_spec["name"],
        brand_raw_spec=brand_spec["raw_spec"],
    )

    # --- 7-4. Gemini 원본 JSON -> Pydantic 응답 스키마로 검증/매핑 ---
    try:
        image_analysis = ImageAnalysis(**gemini_result["image_analysis"])
        brand_config = BrandConfig(**gemini_result["brand_config"])
        generated_copies = GeneratedCopies(**gemini_result["generated_copies"])
    except (KeyError, TypeError) as e:
        logger.error(f"[SCHEMA MAPPING ERROR] gemini_result={gemini_result}")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini 응답 구조가 예상 스키마와 일치하지 않습니다: {str(e)}",
        )

    return GenerateCopyResponse(
        brand_id=brand_id,
        image_analysis=image_analysis,
        brand_config=brand_config,
        generated_copies=generated_copies,
    )


# ------------------------------------------------------------------------------
# 7-1. 보조 엔드포인트: POST /api/chat/refine (디스크립션 1개 길이 조정)
# ------------------------------------------------------------------------------

@app.post(
    "/api/chat/refine",
    response_model=RefineCopyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청 (direction 값 오류 등)"},
        404: {"model": ErrorResponse, "description": "존재하지 않는 브랜드 ID"},
        502: {"model": ErrorResponse, "description": "Gemini API 호출/파싱 오류"},
    },
    summary="이미지 재업로드 없이 디스크립션 1개를 더 짧게/길게 재작성",
)
async def refine_copy(payload: RefineCopyRequest):
    """
    [UX 시나리오 매핑]
    - 프론트엔드는 디스크립션 카드의 '더 짧게'/'더 길게' 버튼 클릭 시
      이 엔드포인트를 호출합니다. 이미지는 다시 보내지 않고,
      이미 화면에 표시된 디스크립션 텍스트 1개만 보냅니다.
    - '되돌리기'는 백엔드 호출 없이, 프론트엔드가 직전 버전을 기억해뒀다가
      그대로 복원하는 방식으로 처리합니다 (서버는 항상 최신 1개만 압니다).
    """
    if payload.direction not in ("shorter", "longer"):
        raise HTTPException(
            status_code=400,
            detail=f"direction은 'shorter' 또는 'longer'여야 합니다. 받은 값: '{payload.direction}'",
        )
    if not payload.original_copy.strip():
        raise HTTPException(status_code=400, detail="original_copy가 비어있습니다.")

    brand_spec = get_brand_spec(payload.brand_id)

    logger.info(
        f"[REFINE REQUEST] brand_id={payload.brand_id}, direction={payload.direction}"
    )

    refined = call_gemini_for_refine(
        brand_name=brand_spec["name"],
        brand_raw_spec=brand_spec["raw_spec"],
        original_copy=payload.original_copy,
        direction=payload.direction,
    )

    return RefineCopyResponse(refined_copy=refined)


# ------------------------------------------------------------------------------
# 8. 보조 엔드포인트: 챗봇 1단계 '브랜드 퀵버튼' 옵션 목록 제공
# ------------------------------------------------------------------------------

@app.get("/api/chat/brands", summary="챗봇 퀵버튼에 표시할 브랜드 목록 조회")
async def list_brands():
    """
    프론트엔드 챗봇 UI가 최초 진입 시 호출하여,
    하단 퀵버튼(광고주 브랜드 선택지)을 동적으로 렌더링하기 위한 엔드포인트.
    """
    return {
        "brands": [
            {"brand_id": key, "brand_name": value["name"]}
            for key, value in BRAND_LINGUISTIC_SPEC.items()
        ]
    }


# ------------------------------------------------------------------------------
# 9. 헬스체크
# ------------------------------------------------------------------------------

@app.get("/health", summary="서버 상태 확인용 헬스체크")
async def health_check():
    return {"status": "ok", "model": GEMINI_MODEL_NAME}


# ------------------------------------------------------------------------------
# 로컬 실행용 (uvicorn main:app --reload 권장하지만, python main.py로도 실행 가능)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
