use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;

fn main() {
    let iris_data = load_dataset();
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );
    let y = iris_data.target;
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    // Create a logistic regression model
    let mut model = LogisticRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    let y_pred = model.predict(&x_test).unwrap();
    let accuracy = accuracy(&y_test, &y_pred);

    println!("Accuracy: {}", accuracy);
}