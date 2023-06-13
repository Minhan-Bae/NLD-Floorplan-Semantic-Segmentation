import numpy as np

def collate_fn(samples):
    """
    입력된 샘플들을 배치로 묶어주는 collate 함수 구현

    Args:
        samples (List[Any]): 개별 샘플들의 리스트. 각 샘플은 모델에 입력될 데이터를 포함합니다.

    Returns:
        Any: 배치로 묶인 데이터
    """
    # 개별 샘플들을 추출하여 리스트에 저장
    data = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # 데이터를 적절한 형태로 변환 (예: NumPy 배열)
    data = np.array(data)
    labels = np.array(labels)

    # 배치로 묶인 데이터 반환
    return data, labels
