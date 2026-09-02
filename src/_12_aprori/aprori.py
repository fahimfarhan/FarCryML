import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def main():
    # Each row is one transaction
    transactions = [
        ["bread", "milk"],
        ["bread", "butter", "milk"],
        ["bread", "butter"],
        ["milk", "eggs"],
        ["bread", "milk", "eggs"],
        ["bread", "butter", "milk"],
        ["milk", "butter"],
        ["bread", "butter", "eggs"],
    ]

    # Convert transactions into one-hot encoded DataFrame
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)

    df = pd.DataFrame(
        encoded,
        columns=encoder.columns_
    )

    print("Transaction data:")
    print(df)

    # Find frequent itemsets
    frequent_itemsets = apriori(
        df,
        min_support=0.3,
        use_colnames=True
    )

    print("\nFrequent itemsets:")
    print(frequent_itemsets)

    # Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.7
    )

    print("\nAssociation rules:")
    print(
        rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ]
    )


if __name__ == "__main__":
    main()