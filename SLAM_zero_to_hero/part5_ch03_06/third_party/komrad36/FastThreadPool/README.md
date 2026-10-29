# FastThreadPool
A super fast lock-free (though not always wait-free) thread pool. Could probably be faster with block queues and should probably sleep instead of spin at least after a while if no work is available - TODO me, I suppose.
