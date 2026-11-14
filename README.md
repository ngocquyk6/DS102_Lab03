Lab03_DS102

Đề bài 1

  Use the Numpy library only to construct the Linear Regression model.
  Train and Evaluate that Linear Regression model on the Forest Fires dataset.

Ý tưởng thực hiện
  Sử dụng lớp `LinearRegression` tự cài đặt trong `Linear_regression.py`.
  Ước lượng trọng số bằng công thức **Normal Equation**:

  $$
  \mathbf{w} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
  $$

  Công thức trên yêu cầu ma trận $\mathbf{X}^{T}\mathbf{X}$ khả nghịch. Phiên bản bài thực hành sử dụng mô hình **không có intercept**.
  Huấn luyện trên tập train, dự đoán trên tập test.
  Đánh giá kết quả bằng **Root Mean Squared Error (RMSE)**:

  $$
  RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
  $$

Đề bài 2

  Standardize the data so that their mean is 0 and their variance is 1.
  Compare the results of Linear Regression model when being trained on the original and standardized data.

Ý tưởng thực hiện

  Xây dựng hai trường hợp thực nghiệm: **trước chuẩn hóa** và **sau chuẩn hóa**.
  Chuẩn hóa các đặc trưng đầu vào theo Z-score:

  $$
  z = \frac{x-\mu}{\sigma}
  $$

  Với cách triển khai chuẩn, tính $\mu$ và $\sigma$ **chỉ trên tập train**, sau đó dùng cùng các tham số để biến đổi tập test; không chuẩn hóa nhãn `area`.
  Huấn luyện Linear Regression trên hai phiên bản dữ liệu; so sánh **RMSE** và in mean/variance của tập train đã chuẩn hóa.

Đề bài 3

  Perform feature engineering to remove columns that are linearly dependent (colinear) with other columns in the dataset.
  Evaluate the Linear Regression model when beeing trained on the original data and preprocessed data.

Ý tưởng thực hiện

  Tính ma trận tương quan Pearson giữa các đặc trưng của **tập train** để khảo sát mối liên hệ tuyến tính.
  Trong phiên bản bài làm hiện tại, xem xét tương quan của `DMC` với các cột khác và thử **loại cột `DMC`** ở cả tập train lẫn test.
  Huấn luyện hai mô hình tương ứng **giữ nguyên đặc trưng** và **loại `DMC`**; so sánh RMSE.

Đề bài 4

  Implement the Linear Regression model using Machine Learning libraries (Scikit Learn or Skorch).
  Compare the results of Linear Regression model constructed manually and from Machine Learning libraries.

Ý tưởng thực hiện

  Huấn luyện **mô hình 1:** `LinearRegression` tự cài đặt với NumPy.
  Huấn luyện **mô hình 2:** `sklearn.linear_model.LinearRegression(fit_intercept=False)`.
  Hai mô hình dùng **cùng tập train/test** và cùng thiết lập **không có intercept**, bảo đảm so sánh nhất quán.
  Dự đoán trên tập test và so sánh **RMSE** của hai mô hình.

