import caffe
import numpy as np

H = np.array([[1,0,0],[0,7.0,0],[0,0,1.5]],dtype = 'f4')
blob = caffe.io.array_to_blobproto( H.reshape( (1,1,3,3) ) )
with open( 'infogainH.binaryproto', 'wb' ) as f :
	f.write( blob.SerializeToString() )