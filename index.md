---
title: "Applied Unsupervised Machine Learning"
date: today
number-sections: false
---

<!--
::: {.callout-warning}
This course is under construction. None of the information should be used
until this message has been removed.
:::
-->

## Overview 

This course on unsupervised learning provides a systematic introduction to dimensionality reduction and clustering techniques. The course covers fundamental concepts of unsupervised learning and data normalization, then progresses through the practical applications of Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE), and hierarchical clustering algorithms.

The course emphasizes both theoretical understanding and hands-on application, teaching students to recognize when different techniques are appropriate and when they may fail. A key learning objective is understanding the limitations of linear methods like PCA. Students learn to evaluate the performance of unsupervised learning methods across diverse data types, with the ultimate goal of generating meaningful hypotheses for further research.

::: {.callout-tip}
### Learning Objectives

- The learning objectives and course outline are detailed below.
<!--
- These describe concepts the learners should grasp and techniques they should be able to use by the end of the course.
- You can think of these as completing the phrase "after this course, the participant should be able to..."
- They are not supposed to be as detailed as the learning objectives of each section, but more high-leve;
-->
:::



### Session 1 (half‑day) *Introduction to unsupervised learning*

1. **Introduction to unsupervised learning and normalization:**  
   Understand the fundamental principles of unsupervised learning and recognize the role that data normalization plays in preparing datasets for analysis.

2. **Why normalization is required:**  
   Explain why normalization is necessary to ensure that features with different scales do not unduly influence unsupervised learning algorithms.

3. **Why dimensionality reduction is required**
   
   Why you need dimensionality reduction. 


5. **Basics of dimensionality reduction:**  
   Describe the core concepts of dimensionality reduction. Then describe Principal Component Analysis (PCA), including how it reduces dimensionality by identifying directions of maximum variance.

6. **Evaluating unsupervised learning results**

   How to check the performance and quality of your unsupervised learning results.

---

### Session 2 (half‑day) *Basics of dimensionality reduction*

1. **Basic applications of PCA:**  
   Apply PCA to real datasets, interpret the resulting principal components, and discuss how these components can reveal underlying structure.

2. **Curse of dimensionality:**  
   Explain the concept of the curse of dimensionality and its implications for the performance and interpretability of clustering and dimensionality‑reduction algorithms.

3. **PCA and t‑SNE:**  
   Compare and contrast PCA and t‑Distributed Stochastic Neighbor Embedding (t‑SNE) as two popular techniques for dimensionality reduction and data visualization.

4. **Basics of t‑SNE:**  
   Explain how t‑SNE projects high‑dimensional data into two or three dimensions while preserving local similarities between points.

5. **Applications to data:**  
   Demonstrate the use of both PCA and t‑SNE on sample datasets to visualize clustering tendencies and uncover hidden patterns.


---

### Session 3 (half‑day) *Basics of Clustering*

1. **Clustering:**  
   Define clustering in the context of unsupervised learning and outline its importance in discovering groupings within data.

2. **Basics of k‑means:**  
   Describe the k‑means clustering algorithm, including how cluster centroids are initialized and updated to minimize within‑cluster variance.

3. **Basics of hierarchical clustering:**  
   Explain the steps of hierarchical clustering, heatmaps, agglomerative approaches, and interpret dendrograms.

4. **Deciding on your clustering approach:**  
   Situations in which you would want to apply hierarchical clustering. Discuss specific use cases: such as when the number of clusters is unknown or when a tree‑based representation is desired—where hierarchical clustering is advantageous.

---

### Session 4 (half‑day) *Practical applications (hands-on)*

1. **When *not* to apply PCA and t‑SNE:**  
   Identify situations where PCA or t‑SNE may produce misleading results or be computationally infeasible, and propose alternative strategies.


2. **Practical applications:**  
   Explore real‑world scenarios where unsupervised learning methods provide actionable insights across various domains.

3. **Practical applications of PCA, t‑SNE and hierarchical clustering to biological data:**  
   Apply PCA, t‑SNE, and hierarchical clustering to biological datasets (e.g., gene expression or single‑cell data), interpret the results, and discuss biological insights gained.

4. **Evaluating unsupervised learning methods**   
   How to evaluate these techniques on different kinds of data (single-cell data, electronic healthcare records, social sciences data): these are used to generate hypotheses. Motivations for next steps.






### Target Audience

Students who have some basic familiarity with Python. There are no prerequisites for knowledge of biology or statistics. The course is designed for those who want to learn how to apply unsupervised machine learning techniques to real-world datasets.


### Prerequisites

Basic familiarity with Python. Course webpage is here: 
[Introduction to Python](https://cambiotraining.github.io/data-analysis-in-r-and-python/)

<!-- Training Developer note: comment the following section out if you did not assign levels to your exercises -->
### Exercises

Exercises in these materials are labelled according to their level of difficulty:

| Level | Description |
| ----: | :---------- |
| {{< fa solid star >}} {{< fa regular star >}} {{< fa regular star >}} | Exercises in level 1 are simpler and designed to get you familiar with the concepts and syntax covered in the course. |
| {{< fa solid star >}} {{< fa solid star >}} {{< fa regular star >}} | Exercises in level 2 combine different concepts together and apply it to a given task. |
| {{< fa solid star >}} {{< fa solid star >}} {{< fa solid star >}} | Exercises in level 3 require going beyond the concepts and syntax introduced to solve new problems. |


## Citation & Authors

Please cite these materials if:

- You adapted or used any of them in your own teaching.
- These materials were useful for your research work. For example, you can cite us in the methods section of your paper: "We carried our analyses based on the recommendations in _YourReferenceHere_".

<!-- 
This is generated automatically from the CITATION.cff file. 
If you think you should be added as an author, please get in touch with us.
-->

{{< citation CITATION.cff >}}


## Acknowledgements

<!-- if there are no acknowledgements we can delete this section -->
- We thank Martin van Rongen, Vicki Hodgson, Hugo Tavares, Paul Fannon, Matt Castle and the Bioinformatics Facility Training Team for their support and guidance.
- [Introduction to Statistical Learning in Python](https://www.statlearning.com/)
<!--
- List any other sources of materials that were used.
- Or other people that may have advised during the material development (but are not authors).
-->