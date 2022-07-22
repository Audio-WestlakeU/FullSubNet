def transform_pesq_range(pesq_score):
    """Transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]"""
    return (pesq_score + 0.5) / 5
