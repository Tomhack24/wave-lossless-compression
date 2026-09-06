import numpy as np


# 予測残差変換

def raw_encode(block):
    return block.astype(np.int16)


def raw_decode(residual):
    return residual.astype(np.uint8)


def diff1_encode(block):
    values = block.astype(np.int16)
    residual = np.empty(len(values), dtype=np.int16)
    #空箱を用意
    residual[0] = values[0]
    residual[1:] = values[1:] - values[:-1]
    return residual


def diff1_decode(residual):
    values = np.cumsum(residual.astype(np.int16))
    #np.cumsumは先頭からの累積和をとる関数らしい。
    
    return values.astype(np.uint8)


def linear2_encode(block):
    values = block.astype(np.int16)
    n = len(values)
    residual = np.empty(n, dtype=np.int16)

    residual[0] = values[0]
    if n == 1:
        return residual
    residual[1] = values[1]

    for i in range(2, n):
        prediction = 2 * values[i - 1] - values[i - 2]
        residual[i] = values[i] - prediction

    return residual


def linear2_decode(residual):
    n = len(residual)
    values = np.empty(n, dtype=np.int16)

    values[0] = residual[0]
    if n == 1:
        return values.astype(np.uint8)
    values[1] = residual[1]

    for i in range(2, n):
        prediction = 2 * values[i - 1] - values[i - 2]
        values[i] = prediction + residual[i]

    return values.astype(np.uint8)


def encode_residual(block, predictor):
    if predictor == "raw":
        return raw_encode(block)
    elif predictor == "diff1":
        return diff1_encode(block)
    elif predictor == "linear2":
        return linear2_encode(block)


def decode_residual(residual, predictor):
    if predictor == "raw":
        return raw_decode(residual)
    elif predictor == "diff1":
        return diff1_decode(residual)
    elif predictor == "linear2":
        return linear2_decode(residual)


#　ライス符号

def zigzag_encode(values):
    values = values.astype(np.int64)
    result = np.where(values >= 0, 2 * values, -2 * values - 1)
    return result.astype(np.uint64)


def zigzag_decode(values):
    values = values.astype(np.int64)
    result = np.where(values % 2 == 0, values // 2, -(values + 1) // 2)
    return result


def best_k(unsigned_values, max_k=16):
    n = len(unsigned_values)

    best_k_value = 0
    best_cost = None
    for k in range(max_k + 1):
        cost = n * (k + 1) + int((unsigned_values >> np.uint64(k)).sum())
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_k_value = k

    return best_k_value, best_cost


def encode_block_bits(unsigned_values, k):
    bits = []
    for v in unsigned_values:
        v = int(v)
        q = v >> k
        bits.extend([1] * q)
        bits.append(0)
        for i in range(k - 1, -1, -1):
            bits.append((v >> i) & 1)
    return np.array(bits, dtype=np.uint8)


class BlockDecoder:
    def __init__(self, payload):
        self.bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        self.pos = 0

    def decode(self, count, k):
        values = np.empty(count, dtype=np.uint64)
        for i in range(count):
            q = 0
            while self.bits[self.pos] == 1:
                q += 1
                self.pos += 1
            self.pos += 1

            remainder = 0
            for _ in range(k):
                remainder = (remainder << 1) | int(self.bits[self.pos])
                self.pos += 1

            values[i] = (q << k) | remainder
        return values




def compress(data, block_size, predictor):
    n = len(data)
    metadata = bytearray()
    bit_chunks = []

    for start in range(0, n, block_size):
        block = data[start:start + block_size]

        residual = encode_residual(block, predictor)
        unsigned = zigzag_encode(residual)
        k, _ = best_k(unsigned)

        metadata.append(k)
        bit_chunks.append(encode_block_bits(unsigned, k))

    all_bits = np.concatenate(bit_chunks)
    packed = np.packbits(all_bits).tobytes()

    payload = bytes(metadata) + packed

    original_bytes = n
    compressed_bytes = len(payload)
    ratio = original_bytes / compressed_bytes
    bits_per_sample = compressed_bytes * 8 / original_bytes

    stats = {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": ratio,
        "bits_per_sample": bits_per_sample,
    }
    return payload, stats


def decompress(payload, n, block_size, predictor):
    n_blocks = (n + block_size - 1) // block_size
    metadata = payload[:n_blocks]

    decoder = BlockDecoder(payload[n_blocks:])
    output = np.empty(n, dtype=np.uint8)

    block_index = 0
    for start in range(0, n, block_size):
        length = min(block_size, n - start)
        k = metadata[block_index]

        unsigned = decoder.decode(length, k)
        residual = zigzag_decode(unsigned).astype(np.int16)
        output[start:start + length] = decode_residual(residual, predictor)

        block_index += 1

    return output


def run_whole_data_experiment(data):
    print("=== データ全体をそのまま圧縮 (表1) ===")
    for predictor in ("raw", "diff1", "linear2"):
        payload, stats = compress(data, block_size=len(data), predictor=predictor)
        restored = decompress(payload, len(data), len(data), predictor)
        ok = np.array_equal(data, restored)
        print(f"{predictor:8s} ratio={stats['ratio']:.3f}  bit/sample={stats['bits_per_sample']:.3f}  可逆性={'OK' if ok else 'NG'}")
    print()


def run_block_size_experiment(data):
    print("=== ブロックサイズを変えた補足実験 (図4) ===")
    block_sizes = [256, 1024, 4096, 16384, 65536]
    for predictor in ("raw", "diff1", "linear2"):
        ratios = []
        for block_size in block_sizes:
            payload, stats = compress(data, block_size=block_size, predictor=predictor)
            restored = decompress(payload, len(data), block_size, predictor)
            assert np.array_equal(data, restored)
            ratios.append(stats["ratio"])
        ratio_text = ", ".join(f"{r:.2f}" for r in ratios)
        print(f"{predictor:8s} block_size={block_sizes} -> ratio=[{ratio_text}]")
    print()


def run_time_segment_experiment(data):
    print("=== 1秒ごとの圧縮率の時間推移 (図5) ===")
    sample_rate = 65536
    segment_size = sample_rate
    n_segments = len(data) // segment_size

    for predictor in ("raw", "diff1", "linear2"):
        ratios = []
        for i in range(n_segments):
            segment = data[i * segment_size:(i + 1) * segment_size]
            payload, stats = compress(segment, block_size=4096, predictor=predictor)
            restored = decompress(payload, len(segment), 4096, predictor)
            assert np.array_equal(segment, restored)
            ratios.append(stats["ratio"])
        print(f"{predictor:8s} min={min(ratios):.3f} max={max(ratios):.3f} mean={np.mean(ratios):.3f}")
    print()


if __name__ == "__main__":
    with open("data/wave_2026.dat", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    run_whole_data_experiment(data)
    run_block_size_experiment(data)
    run_time_segment_experiment(data)
