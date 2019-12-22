import numpy
import matplotlib.image as mpimg
import random
import sys

class ImageBlock:
    def __init__(self, image_representation, tss, trb, tcb):
        self.usingImage = image_representation
        self.tss = tss
        self.trb = trb
        self.tcb = tcb

    def CopyBlock(self, block):
        self.usingImage.content[self.trb : self.trb + self.tss, self.tcb : self.tcb + self.tss] = block.usingImage.content[block.trb : block.trb + block.tss, block.tcb : block.tcb + block.tss]
    
    def CopyBlock1(self, block, ffs):
        org = self.usingImage.content[self.trb : self.trb + self.tss, self.tcb : self.tcb + self.tss]
        target = block.usingImage.content[block.trb : block.trb + block.tss, block.tcb : block.tcb + block.tss]
        self.usingImage.content[self.trb : self.trb + self.tss, self.tcb : self.tcb + self.tss] = org*(1 - ffs.content) + target *  ffs.content
        return

    def RefPixel(self, r, c, cIdx) :
        return self.usingImage.RefPixel(r + self.trb, c + self.tcb, cIdx)


    def generateCutFlags(self, ib, ols):
        errorSurface = self.usingImage.distanceMetric2(ib, self.trb, self.tcb, ols)
        curFlags = NumpyMatrixWrapper(ib.tss, ib.tss, 1)
        for r in range(curFlags.Rows()):
            for c in range(curFlags.Columns()):
                curFlags.SetPixel(r, c, 0, 1)

            if self.trb != 0:
                tis =  NumpyMatrixWrapper(ols, self.tss, 1)
                crs = NumpyMatrixWrapper(0,0,0)
                crs.Clone(errorSurface.content)
                for w_fake in range(self.tss - 1):
                    c = w_fake + 1
                    for h1 in range(ols):
                        rows = [h1 - 1, h1, h1 + 1]
                        minI = h1
                        for i in range(3):
                            if rows[i] == minI:
                                continue
                            if rows[i] < 0:
                                continue
                            if rows[i] >= ols:
                                continue
                            if crs.RefPixel(rows[i], c - 1, 0) < crs.RefPixel(minI, c - 1, 0): 
                                minI = rows[i]
                        crs.IncPixel(h1, c, 0, crs.RefPixel(minI, c - 1, 0))
                        tis.SetPixel(h1, c, 0, minI)

                minI = 0
                for i in range(ols):
                    if crs.RefPixel(i, self.tss - 1, 0) < crs.RefPixel(minI,self.tss - 1, 0):
                        minI = i

                sis = [0 for ii in range(self.tss)]
                for i in range(self.tss):
                    currentI = self.tss - 1 - i
                    sis[currentI] = minI
                    minI = tis.RefPixel(minI, currentI, 0)

                for c in range(self.tss):
                    for h1 in range(ols):
                        if h1 > sis[c]:
                            break
                        else:
                            curFlags.SetPixel(h1, c, 0, 0)

            if self.tcb != 0:
                tis =  NumpyMatrixWrapper(self.tss, ols, 1)
                crs = NumpyMatrixWrapper(0,0,0)
                crs.Clone(errorSurface.content)
                for h1_fake in range(self.tss - 1):
                    h1 = h1_fake + 1
                    for c in range(ols):
                        cols = [ c - 1, c, c + 1 ]
                        minI = c
                        for i in range(3):
                            if cols[i] == minI:
                                continue
                            if cols[i] < 0:
                                continue
                            if cols[i] >= ols:
                                continue
                            if crs.RefPixel(h1 - 1, cols[i], 0) < crs.RefPixel(h1 - 1, minI, 0): 
                                minI = cols[i]
                            
                        crs.IncPixel(h1, c, 0, crs.RefPixel(h1 - 1, minI, 0))
                        tis.SetPixel(h1, c, 0, minI)

                minI = 0
                for i in range(ols):
                    if crs.RefPixel(self.tss - 1, i, 0) < crs.RefPixel(self.tss - 1,minI, 0):
                        minI = i
                sis = [0 for ii in range(self.tss)]
                for i in range(self.tss):
                    currentI = self.tss - 1 - i
                    sis[currentI] = minI
                    minI = tis.RefPixel(currentI, minI, 0)

                for h1 in range(self.tss):
                    for c in range(ols):
                        if c > sis[h1]:
                            break
                        else:
                            curFlags.SetPixel(h1, c, 0, 0)
        return curFlags

class ErrorInfo :
    def __init__(self, f, s) :
        self.first = f
        self.second = s

class NumpyMatrixWrapper:
    content = None
    # mat is a numpy matrix
    def __init__(self, r, c, es):
        if r*c*es != 0:
            self.content = numpy.zeros((r, c, es)).astype(numpy.intc)
        else:
            self.content = None
    
    def Rows(self):
        return self.content.shape[0]

    def Columns(self):
        return self.content.shape[1]

    def Clone(self, mat):
        if mat is None:
            self.content = None
        else:
            self.content = numpy.copy(mat)
            if len(self.content.shape) == 2:
                self.content.resize((self.content.shape[0], self.content.shape[1], 1))
    
    def RefPixel(self, r, c, cIdx):
        return self.content[(r,c,cIdx)]

    def SetPixel(self, r, c, cIdx, value):
        self.content[(r,c,cIdx)] = value

    def IncPixel(self, r, c, cIdx, value):
        self.content[(r,c,cIdx)] += value

    def distanceMetric1(self, ib, o_r, oc, ols):
        return numpy.sum(self.distanceMetric2(ib, o_r, oc,ols).content)

    def distanceMetric2(self, ib, o_r, oc, ols) : 
        tss = ib.tss
        result = NumpyMatrixWrapper(0, 0, 0)
        outputBlock = ImageBlock(self, tss, o_r, oc)

        input_matrix = ib.usingImage.content[ib.trb : ib.trb + ib.tss, ib.tcb : ib.tcb + ib.tss]
        output_matrix = outputBlock.usingImage.content[outputBlock.trb : outputBlock.trb + outputBlock.tss, outputBlock.tcb : outputBlock.tcb + outputBlock.tss]
        diff_matrix = numpy.power(input_matrix - output_matrix, 2)
        diff_matrix = numpy.sum(diff_matrix, axis=2)
        diff_matrix[ols : tss, ols : tss] = numpy.zeros((tss-ols, tss-ols))
        if oc == 0:
            diff_matrix[ols:tss, 0:ols]  = numpy.zeros((tss-ols, ols))
        if o_r == 0:
            diff_matrix[0:ols, ols:tss]  = numpy.zeros((ols, tss-ols))
        result.Clone(diff_matrix)
        return result

def ReadImage(filePath):
    src = mpimg.imread(filePath)
    image = NumpyMatrixWrapper(0, 0, 0)
    image.Clone(src)
    return image

def WriteImage(image, filePath):
    mpimg.imsave(filePath, image.content.astype(numpy.uint8))

class ImageProcessor :

    def __init__(self, ir, tss, ooh, oow) : 
        self.inputImg = ir
        self.tss = tss
        mts = min(ir.Rows(), ir.Columns())
        if mts < self.tss :
            self.tss = mts
        self.ols = int(self.tss / 3)
        if self.ols <= 0: 
            self.ols = 1
        self.oor = ooh
        self.ooc = oow
        I1s = 0
        I2s = 0
        self.op = self.tss - self.ols

        if self.tss < self.ooc :
            tv = (self.ooc - self.tss) / 1.0 / self.op
            I1s = int( tv )
            if tv != int(tv):
                I1s = I1s + 1

        if self.tss < self.oor :
            tv = (self.oor - self.tss) / 1.0 / self.op
            I2s = int( tv)
            if tv != int(tv) : 
                I2s = I2s + 1

        self.o_r = self.tss + I2s * self.op
        self.oc = self.tss + I1s * self.op
        self.oi = NumpyMatrixWrapper(self.o_r, self.oc, 3)
        self.ir = self.inputImg.Rows()
        self.ic = self.inputImg.Columns()

    def  process(self): 
        t_s_s = 0
        blockH = 0
        while blockH + self.tss <= self.o_r:
            t_s_s = t_s_s + 1
            blockH = blockH + self.op
        currentStepIdx = 0
        for loopI in range(t_s_s):
            blockH = loopI * self.op
            blockW = 0
            while blockW + self.tss <= self.oc :
                outputBlock =  ImageBlock(self.oi, self.tss, blockH, blockW)
                blockHeightCount = self.ir - self.tss + 1
                blockWidthCount = self.ic - self.tss + 1
                if blockH == 0 and blockW == 0:
                    usingBlockH = random.randint(0, blockHeightCount-1)
                    usingBlockW = random.randint(0, blockWidthCount-1)
                    ib =  ImageBlock(self.inputImg, self.tss, usingBlockH, usingBlockW)
                    outputBlock.CopyBlock(ib)
                else:
                    errArr = []
                    for ibHBegin in range(blockHeightCount):
                        for ibWBegin in range(blockWidthCount):
                            loc = ibHBegin * blockWidthCount + ibWBegin
                            err = self.oi.distanceMetric1( ImageBlock(self.inputImg, self.tss,ibHBegin, ibWBegin), blockH, blockW, self.ols)
                            errArr.append( ErrorInfo(loc, err))
                    minErr = errArr[0].second
                    for i in range(len(errArr)):
                        if errArr[i].second < minErr :
                            minErr = errArr[i].second
                    te = minErr * 1.1
                    ca = []
                    for i in range(len(errArr)):
                        if errArr[i].second <= te:
                            ca.append(i)

                    selectI = ca[random.randint(0, len(ca)-1)]
                    usingBlockH = int(selectI / blockWidthCount)
                    usingBlockW = int(selectI % blockWidthCount)
                    ib =  ImageBlock(self.inputImg, self.tss, usingBlockH, usingBlockW)
                    outputBlock.CopyBlock1(ib, outputBlock.generateCutFlags(ib, self.ols))

                blockW = blockW + self.op
            currentStepIdx = currentStepIdx + 1
            print("percent: %.2f%%(%d/%d)" % (currentStepIdx / 1.0 / t_s_s * 100.0, currentStepIdx, t_s_s))

        correct_size_image =  NumpyMatrixWrapper(self.oor, self.ooc, 3)
        for r in range(correct_size_image.Rows()):
            for c in range(correct_size_image.Columns()):
                for i in range(3):
                    correct_size_image.SetPixel(r, c, i, self.oi.RefPixel(r, c, i))
        return correct_size_image



if __name__ == '__main__':
    input_path = ""
    output_path = ""
    texton_size = 0
    output_size = 0
    argv = sys.argv
    for i in range(len(argv)):
        if argv[i] == "-input" and (i + 1) < len(argv): 
            input_path = argv[i + 1]
        elif argv[i] == "-texton-size" and (i + 1) < len(argv) :
            texton_size = int(argv[i + 1])
        elif argv[i] == "-output-size" and (i + 1) < len(argv) :
            output_size = int(argv[i + 1])
        elif (argv[i] == "-output" and (i + 1) < len(argv)) :
            output_path = argv[i + 1]
    input_image = ReadImage(input_path)
    image_analysis = ImageProcessor(input_image, texton_size, output_size, output_size)
    out_image = image_analysis.process()
    WriteImage(out_image, output_path)