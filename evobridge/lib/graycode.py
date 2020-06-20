def grayToStdbin(genotype):
    assert type(genotype) == bytearray

    def transform(gray):
        bin = 0
        while gray > 0:
            d = gray - 1
            bin ^= gray
            bin ^= d
            gray &= d

        return bin

    return bytearray(map(transform, genotype))
