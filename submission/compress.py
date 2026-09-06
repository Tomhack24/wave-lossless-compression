import struct
import numpy as np


# ---------- 予測残差変換 ----------

def raw_encode(block):
    return block.astype(np.int16)


def raw_decode(residual):
    return residual.astype(np.uint8)


def diff1_encode(block):
    values = block.astype(np.int16)
    residual = np.empty(len(values), dtype=np.int16)
    residual[0] = values[0]
    residual[1:] = values[1:] - values[:-1]
    return residual


def diff1_decode(residual):
    values = np.cumsum(residual.astype(np.int32))
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
    values = np.empty(n, dtype=np.int32)

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


# ---------- Rice符号 ----------

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


# ---------- コーデック（ブロック分割・手法選択） ----------

MAGIC = b"WLC1"
PREDICTOR_TO_ID = {"raw": 0, "diff1": 1, "linear2": 2}
ID_TO_PREDICTOR = {v: k for k, v in PREDICTOR_TO_ID.items()}
HEADER_FORMAT = ">4sII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def compress(data, block_size, predictors=("raw", "diff1", "linear2")):
    n = len(data)
    metadata = bytearray()
    bit_chunks = []
    predictor_counts = {name: 0 for name in predictors}

    for start in range(0, n, block_size):
        block = data[start:start + block_size]

        best_cost = None
        for predictor in predictors:
            residual = encode_residual(block, predictor)
            unsigned = zigzag_encode(residual)
            k, cost = best_k(unsigned)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_predictor = predictor
                best_kv = k
                best_unsigned = unsigned

        predictor_counts[best_predictor] += 1
        metadata.append((PREDICTOR_TO_ID[best_predictor] << 6) | best_kv)
        bit_chunks.append(encode_block_bits(best_unsigned, best_kv))

    all_bits = np.concatenate(bit_chunks)
    packed = np.packbits(all_bits).tobytes()

    header = struct.pack(HEADER_FORMAT, MAGIC, n, block_size)
    payload = header + bytes(metadata) + packed

    original_bytes = n
    compressed_bytes = len(payload)
    ratio = original_bytes / compressed_bytes
    bits_per_sample = compressed_bytes * 8 / original_bytes

    stats = {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": ratio,
        "bits_per_sample": bits_per_sample,
        "predictor_counts": predictor_counts,
    }
    return payload, stats


def decompress(payload):
    magic, n, block_size = struct.unpack_from(HEADER_FORMAT, payload, 0)

    offset = HEADER_SIZE
    n_blocks = (n + block_size - 1) // block_size
    metadata = payload[offset:offset + n_blocks]
    offset += n_blocks

    decoder = BlockDecoder(payload[offset:])
    output = np.empty(n, dtype=np.uint8)

    block_index = 0
    for start in range(0, n, block_size):
        length = min(block_size, n - start)
        meta_byte = metadata[block_index]
        predictor = ID_TO_PREDICTOR[meta_byte >> 6]
        k = meta_byte & 0x3F

        unsigned = decoder.decode(length, k)
        residual = zigzag_decode(unsigned).astype(np.int16)
        output[start:start + length] = decode_residual(residual, predictor)

        block_index += 1

    return output


# ---------- 実行部分 ----------

if __name__ == "__main__":
    with open("data/wave_2026.dat", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    block_size = 16384
    payload, stats = compress(data, block_size)
    restored = decompress(payload)

    print("元データ:", stats["original_bytes"], "bytes")
    print("圧縮後  :", stats["compressed_bytes"], "bytes")
    print("圧縮率  :", round(stats["ratio"], 3))
    print("bit/sample:", round(stats["bits_per_sample"], 3))
    print("予測手法の内訳:", stats["predictor_counts"])
    print("可逆性:", "OK" if np.array_equal(data, restored) else "NG")
