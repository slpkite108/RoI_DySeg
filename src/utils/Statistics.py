from heapq import heappush, heappop
class Statistics:    
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.min_heap = []
        self.max_heap = []
        
    def add_number(self, number):
        
        self.sum += number
        self.count += 1
        self.min = min(self.min, number)
        self.max = max(self.max, number)
        
        if not self.max_heap or number <= -self.max_heap[0]:
            heappush(self.max_heap, -number)
        else:
            heappush(self.min_heap, number)
        
        # Balance the heaps
        if len(self.max_heap) > len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, -heappop(self.min_heap))
    
    def get_sum(self):
        return self.sum
    
    def get_mean(self):
        if self.count == 0:
            return float('nan')
        return self.sum / self.count
    
    def get_median(self):
        if self.count == 0:
            return float('nan')
        
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0])/2.0
        
        return -self.max_heap[0]
    
    def get_min(self):
        return self.min if self.count>0 else float('nan')
    
    def get_max(self):
        return self.max if self.count>0 else float('nan')
