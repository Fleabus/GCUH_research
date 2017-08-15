#class (continuousStreamArray, batch size, step) - Aidan playing around with OO
#class for heatmap

class Datastream:
    dataArray = []

    def __init__(self):
        return

    def setData(self, array):
        self.dataArray = array

    def resetData(self):
        self.dataArray = []

    def retrieveBatches(self, batchSize, step):
        dataLength = len(self.dataArray)
        iters = int((dataLength-batchSize/2)/step)
        halfBatch = int(batchSize/2)
        batchArray = []
        for i in range(halfBatch, iters):
            #print("i:", i, " batchfrom:", i-halfBatch, " batchTo:", i+halfBatch)
            batchArray.append(self.dataArray[i-halfBatch:i+halfBatch])
        return batchArray

test = Datastream()
array = []
for i in range(0, 900):
    array.append(1)
test.setData(array)
batchArray = test.retrieveBatches(360,1)
print(len(batchArray[0]))
