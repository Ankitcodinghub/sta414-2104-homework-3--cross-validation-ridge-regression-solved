# sta414-2104-homework-3--cross-validation-ridge-regression-solved
**TO GET THIS SOLUTION VISIT:** [STA414-2104 Homework 3- Cross validation ridge regression Solved](https://www.ankitcodinghub.com/product/sta414-2104-homework-3-cross-validation-ridge-regression-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;96442&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;STA414-2104 Homework 3- Cross validation ridge regression Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
2. Regression –&nbsp; In this question, you will derive certain properties of linear regres- sion.

2.1. Linear regression –&nbsp; Suppose that 2 Rn⇥m with n m and t 2 Rn, and that t|(, w) ⇠ N (w, 2I). We know that the maximum likelihood estimate wˆ of w is given by

wˆ = (T )1&gt;t.

<ol>
<li>(a) &nbsp;Write the log-likelihood implied by the model above, and compute its gradient w.r.t. w. By setting it equal to 0, derive the above estimator wˆ .</li>
<li>(b) &nbsp;Find the distribution of wˆ , its expectation and covariance matrix.</li>
</ol>
What to submit?

<ol>
<li>a) &nbsp;Log-likelihood, its gradient, and your entire derivation.</li>
<li>b) &nbsp;Use the property in 1.1 c) of multivariate Gaussian random vectors, and find the distribution,
and calculate its expectation and variance.
</li>
</ol>
2.2. Ridge regression and MAP – . Suppose that we have t|(, w) ⇠ N (w, 2I) and we place a normal prior on w|, i.e., w ⇠ N(0,⌧2I). Recall from the first lecture (also in prelim- inaries.pdf) that MAP estimate of w is given as the maximum of the posterior density

wˆMAP =argmax{p(w|,t)/p(t|,w)p(w|)}. w

Here, / notation means proportional to, and is used since we dropped the term p(t|) in the denominator as it doesn’t have w in it, thus it doesn’t contribute to the maximization problem.

Show that the MAP estimate of w given (t, ) in this context is

(2.1) wˆMAP =(&gt;+I)1&gt;t

where = 2/⌧2. What to submit?

a) Submit all your derivations.

3. Cross validation – 40 pts. In this problem, you will write a function that performs K-fold cross validation (CV) procedure to tune the penalty parameter in Ridge regression. CV procedure is one of the most commonly used methods for tuning hyperparameters. In this question, you shouldn’t use the package scikit-learn to perform CV. You should implement all of the below functions yourself. You may use numpy and scipy for basic math operations such as linear algebra, sampling etc.

In class we learned training, test, and validation procedures which assumes that you have enough data and you can set aside a validation set and a test set to use it for assessing the performance of your machine learning algorithm. However in practice, this may be problematic since we may not have enough data. A remedy to this issue is K-fold cross- validation which uses a part of the available data to fit the model, and a di↵erent part to test it. K-fold CV procedure splits the data into K equal-sized parts; for example, when K = 5, the scenario looks like this:

2

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Fig 1: credit: Elements of Statistical Learning

<ol>
<li>We first set aside a test dataset and never use it until the training and parameter tuning procedures are complete. We will use this data for final evaluation. In this question, test data is provided to you as a separate dataset.</li>
<li>CV error estimates the test error of a particular hyperparameter choice. For a particular hyperparameter value, we split the training data into K blocks (See the figure), and for k = 1, 2, …, K we use the k-th block for validation and the remaining K 1 blocks are for training. Therefore, we train and validate our algorithm K times. Our CV estimate for the test error for that particular hyperparameter choice is given by the average validation error across these K blocks.</li>
<li>We repeat the above procedure for several hyperparameter choices and choose the one that provides us with the smalles CV error (which is an estimate for the test error).</li>
</ol>
Below, we will code the above procedure for tuning the regularization parameter in linear regression which is a hyperparameter. Your cross_validation function will rely on 6 short functions which are defined below along with their variables.

<ul>
<li>data is a variable and refers to a (t, ) pair (can be test, training, or validation) where t is the target (response) vector, and is the feature matrix.</li>
<li>model is a variable and refers to the coecients of the trained model, i.e. wˆ .</li>
<li>data_shf = shuffle_data(data) is a function and takes data as an argument and returns its randomly permuted version along the samples. Here, we are considering a uniformly random permutation of the training data. Note that t and need to be permuted the same
way preserving the target-feature pairs.
</li>
<li>data_fold, data_rest = split_data(data, num_folds, fold) is a function that takes
data, number of partitions as num_folds and the selected partition fold as its arguments and returns the selected partition (block) fold as data_fold, and the remaining data as data_rest. If we consider 5-fold cross validation, num_folds=5, and your function splits the data into 5 blocks and returns the block fold (2 {1, 2, 3, 4, 5}) as the validation fold and the remaining 4 blocks as data_rest. Note that data_rest [ data_fold = data, and data_rest \ data_fold = ;.
</li>
<li>model = train_model(data, lambd) is a function that takes data and lambd as its argu- ments, and returns the coecients of ridge regression with penalty level . For simplicity, you may ignore the intercept and use the expression in equation (2.1).</li>
<li>predictions = predict(data, model) is a function that takes data and model as its arguments, and returns the predictions based on data and model.
3
</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
<ul>
<li>error = loss(data, model) is a function which takes data and model as its arguments and returns the average squared error loss based on model. This means if data is composed oft2Rn and2Rn⇥p,andmodeliswˆ,thenthereturnvalueisktwˆk2/n.</li>
<li>cv_error = cross_validation(data, num_folds, lambd_seq) is a function that takes the training data, number of folds num_folds, and a sequence of ’s as lambd_seq as its arguments and returns the cross validation error across all ’s. Take lambd_seq as evenly spaced 50 numbers over the interval (0.02, 1.5). This means cv_error will be a vector of 50 errors corresponding to the values of lambd_seq. Your function will look like:
<pre>     data = shuffle_data(data)
     for i = 1,2,...,length(lambd_seq)
</pre>
<pre>            lambd = lambd_seq(i)
            cv_loss_lmd = 0.
            for fold = 1,2, ...,num_folds
</pre>
<pre>                 val_cv, train_cv = split_data(data, num_folds, fold)
                 model = train_model(train_cv, lambd)
                 cv_loss_lmd += loss(val_cv, model)
</pre>
<pre>            cv_error(i) = cv_loss_lmd / num_folds
     return cv_error
</pre>
Download the dataset from the course webpage hw1_data.zip and place and extract in your

working directory, or note its location file_path. For example, file path could be /Users/yourname/Desktop/
</li>
</ul>
• InPython:

<pre>     import numpy as np
     data_train = {’X’: np.genfromtxt(’data_train_X.csv’, delimiter=’,’),
</pre>
<pre>                   ’t’: np.genfromtxt(’data_train_y.csv’, delimiter=’,’)}
     data_test = {’X’: np.genfromtxt(’data_test_X.csv’, delimiter=’,’),
                  ’t’: np.genfromtxt(’data_test_y.csv’, delimiter=’,’)}
</pre>
Here, the design matrix is loaded as data_??[’X’], and target vector t is loaded as data_??[’t’], where ?? is either train or test.

<ol>
<li>(a) &nbsp;Write the above 6 functions, and identify the correct order and arguments to do cross validation.</li>
<li>(b) &nbsp;Find the training and test errors corresponding to each in lambd_seq. This part does not use the cross_validation function but you may find the other functions helpful.</li>
<li>(c) &nbsp;Plot training error, test error, and 5-fold and 10-fold cross validation errors on the same plot for each value in lambd_seq. What is the value of proposed by your cross validation procedure? Comment on the shapes of the error curves.</li>
</ol>
What to submit?

<ol>
<li>a) &nbsp;The functions your wrote.</li>
<li>b) &nbsp;Report the errors you find.</li>
<li>c) &nbsp;The plot containing 4 curves: i) training ii) test, iii) 5-fold CV iv) 10-fold CV errors, where
x axis is lambda.
</li>
<li>d) &nbsp;Your entire code should be attached to the end of your answers.
4
</li>
</ol>
</div>
</div>
</div>
