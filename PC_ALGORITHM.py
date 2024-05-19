# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# %%
mnist = fetch_openml(data_id=554,as_frame=False)
x, y = mnist.data.astype('float32'), mnist.target.astype('int')
#normalize input range becomes [0.0,1.0]
x/= 255.0

# %%
digits_478=[4,7,8]
indices_478 = np.isin(y, digits_478)
train_x = x[indices_478]
train_y = y[indices_478]
train_x

# %%
train_x=train_x[0:900]
train_y=train_y[0:900]
# print(train_x.dtype,train_x.shape,train_y.shape)
# print(train_y)



# %%
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_x[i].reshape(28,28), cmap=plt.get_cmap('gray'))

# %%
sample_covariance_matrix=np.cov(train_x.T,dtype=np.float32)
print(sample_covariance_matrix.shape,
sample_covariance_matrix.dtype)

# %%
eigenvalue_scov,eigenvector_scov=np.linalg.eigh(sample_covariance_matrix)
print(eigenvalue_scov.shape,
eigenvector_scov.shape)
print(eigenvalue_scov.dtype,
eigenvector_scov.dtype)

# %%
sorted_index = np.argsort(eigenvalue_scov)[::-1]
print(sorted_index)

# %%
eigenvalue_scov_sorted=eigenvalue_scov[sorted_index]
eigenvector_scov_sorted=eigenvector_scov[:,sorted_index]
print(eigenvalue_scov_sorted)


# %%
m_array=[2, 10, 50, 100, 200, 300]

# %%
for m in m_array:
  #MATRIX A of top m eigenvectors
  principal_components=eigenvector_scov_sorted[:,:m].T
  #new feature vector
  y=np.dot(principal_components,train_x.T)
  print(y.shape)
  #reconstruction from m to 784 for visualization
  reconstructed_x=np.dot(principal_components.T,y).T

  # now we will plot first 10 images for each m
  # for plotting it should be in transformed from 784 to 28x28
  images=train_x[2:8].reshape(-1,28,28)
  reconstructed_images=reconstructed_x[2:8].reshape(-1,28,28)

  plt.figure(figsize=(15, 6))
  for i in range(6):
      plt.subplot(2, 6, i + 1)
      plt.imshow(images[i], cmap='gray')
      plt.title(f'Original {train_y[i]}')
      plt.axis('off')

      plt.subplot(2, 6, i + 7)
      plt.imshow(reconstructed_images[i], cmap='gray')
      plt.title(f'Rec {train_y[i]}')
      plt.axis('off')

  plt.suptitle(f'Original and Reconstructed images with new feature vector size {m}')
  plt.show()



# %%
variance=eigenvalue_scov_sorted/np.sum(eigenvalue_scov_sorted)
print(variance.shape,variance.dtype)

# %%
plt.plot(np.arange(1,785),variance,marker='.', linestyle='')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Sum(Variance)')
plt.grid()

# %%
cummulative_sum=np.cumsum(variance)
print(cummulative_sum.shape,cummulative_sum.dtype)

# %%
plt.plot(np.arange(1,785),cummulative_sum,marker='.', linestyle='')

plt.xlabel('Number of Features')
plt.ylabel('Cumulative Sum(Variance)')
plt.grid()

# %%
pca_98=784-len(cummulative_sum[cummulative_sum>0.98])
print("m where variance is 98% is :",pca_98)


