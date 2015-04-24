mport pandas as pd

# Let's read in the data
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data", header=None)

--

df.head(3)

--

df.describe()

--

df[1][0]

--

df = df.convert_objects(convert_numeric=True)

--

df[1][0]

--

df.describe()

--

df=df.interpolate()
# interpolate only works with numerical data
# almost all models require numerical data only except naive bayes

--

df.describe()

--

df.count()

--

df.head(3)

--

#  Now let's turn categorical data into numbers
# Let's see what get_dummies does
pd.get_dummies(df[0])

--

#  Looks like there are some missing values, which are set to '?'
# Precisely 12 rows that have a missing value in column 0
df[df[0] == '?']

--

# Let's go ahead and remove the rows that have missing values in the categorical columns
# since we have no natural imputation we want to apply for missing categorical data
# a.k.a. no good way to fill in those missing values
categorical_columns = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]
for i in categorical_columns:
    df = df[df[i] != '?']

--

# now we're down to 671 rows
df.describe()

--

# Let's see what get_dummies gets us for column 0 this time
# We expect no '?'
pd.get_dummies(df[0]).head()

--

# Great! But we don't want to include all these columns
# otherwise we'll run into multicollinearity, right?
# so we'll drop one column for each of the dummifications

# So let's go through each categorical column
# convert to dummy variable columns
# drop one and replace the original categorical column with the dummy variable columns

count = 0
for i in categorical_columns:
    # all but the first column since we want to avoid multicollinearity
    # and it doesn't matter which dummy column we get rid of
    dummies_data_frame = pd.get_dummies(df[i]).iloc[:, 1:]
    for col in dummies_data_frame.columns:
        # add the dummy column to df, with a unique column name
        df['{}{}'.format(col, count)] = dummies_data_frame[col]
        count += 1

--

# Let's see what df looks like
df.head()

--

# Let's get rid of the original categorical columns
df.drop(df.columns[categorical_columns], axis=1, inplace=True)

--

df.head()

--


