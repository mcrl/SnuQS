from mqt_bench import MQTBench

braket = MQTBench("ghz", "alg", 5)
braket.run(1000)

snuqs = MQTBench("ghz", "alg", 5, backend="snuqs")
snuqs.run(1000)