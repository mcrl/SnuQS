import ghz
circ_braket = ghz.GHZ(3)
circ_braket.run(1000)

circ_snuqs = ghz.GHZ(3, backend='snuqs')
circ_snuqs.run(1000)
