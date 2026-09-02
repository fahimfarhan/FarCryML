from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.svm import SVC

def main():
    z_score_scaler = StandardScaler()
    model = make_pipeline(
        z_score_scaler, # <--- use it like this in sklearn
        SVC(),
    )

    """
    z score / other scalars are only applied on training x, not test x.
    make_pipeline automatically handles it.
    """

    pass

if __name__ == "__main__":
    main()
    pass
