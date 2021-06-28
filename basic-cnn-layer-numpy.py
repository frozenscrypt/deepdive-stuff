import numpy as np



def cnn(img,filter_size,stride,filters_dim,pad=True):
	pad = (filter_size-1)//2
	dim = img.shape[1]
	nf = img.shape[0]
	canvas = np.zeros((nf,dim+2*pad,dim+2*pad))
	cd = canvas.shape[1]
	canvas[:,pad:cd-pad,pad:cd-pad] = img

	fan_in = nf
	fan_out = filters_dim
	limit = np.sqrt(6/(fan_in+fan_out))
	weights = [np.random.uniform(-limit,limit,size=(nf,filter_size,filter_size)) for i in range(filters_dim)]

	res_dim = (canvas.shape[1]-filter_size)//stride + 1
	wn = 0
	output = np.zeros((filters_dim,res_dim,res_dim))
	for weight in weights:
		result = np.zeros((res_dim,res_dim))
		for i in range(0,canvas.shape[1]-filter_size+1,stride):
			for j in range(0,canvas.shape[2]-filter_size+1,stride):
				
				box = canvas[:,i:i+filter_size,j:j+filter_size]
				box_res = np.sum(np.multiply(box,weight))
				result[i,j] = box_res
				
			
		output[wn,:,:] = result
		wn+=1

	return output




print(cnn(np.ones((3,5,5)),3,1,5,True))

