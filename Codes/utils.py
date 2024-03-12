class Gene:
    def __init__(self, chrom, start, end, strand, name):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand
        self.name = name

class TSS:
    def __init__(self, chrom, pos, reliable=100):
        self.chrom = chrom
        self.pos = pos
        self.re = reliable

    def __str__(self):
        return str(self.chrom) + '\t' + str(self.pos) + '\t' + str(self.re)
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
