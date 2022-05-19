pipeline = i.Pipeline([
    ('remove_stop_words', pp.remove_stop_words()),
])

t = pipeline.fit_transform(X_train)
