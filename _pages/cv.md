---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* MSc in Artificial Intelligence and Adaptive Systems, University of Sussex, 2021-2022
* BSc in Computer Science and Artificial Intelligence, University of Sussex, 2018-2021

Work Experience
======
* HyperTunnel, Mar 2023 - Aug 2023, AI Engineer Intern.
  * Developed and optimized swarm intelligence evolutionary algorithms, enhancing the efficiency of surveying robots for underground construction. Dynamic adaptive optimization of multi-objective functions.
  * Implementation of AI-driven solutions for dynamic excavation tasks, improving geological detection and operational precision.
  * Supervisor: Xinghui Tao

* Microsoft, Oct 2021 - Jun 2022, Mentee of EMBRACE Mentoring Program
  * Applying fairness ML algorithms to improve the feedback of products or advertisements among diverse populations.
  * Explore users' preferences and habits with reinforcement learning, adjusting the product's features and interface, to achieve a better user experience.

Research Experience
======
* A*STAR I2R, Aug 2023 - , Research Engineer.
  * Biological neural connectomes modelling (based on Zebrafish and C. elegans), designing DNN according to the network structural connections and optimizing parameters. Analysing and adjusting specific circuits'(pathways) and neuronal data to enhance the model's adaptability and robustness in multiple environments.
  * Constructing virtual scenes based on Unity 3D (automatically randomly generated) as a training set for visual algorithms. Performance optimization, model training, parameter fine turning for continuous learning networks.
  
* Alonso lab, University of Sussex, Nov 2023 - Mar 2023, Visiting research Student.
  * Participated in the construction of a fly larva hatching platform and data science analysis. Employed dynamic image segmentation methods to acquire time-series data of larvae in dishes, detecting key information such as hatching points, frequency, and activity levels, etc.
  * Development of a tool for large-scale automatic analysis of video recording data (key point detection) to constructing 3D posture data, based on DeepLabCut platform.

Skills
======
* Programming Languages
  * Proficient in Python, MATLAB, Java, and R
  * Experienced with C, C++, C#, JavaScript, Unity3D
* Software and Data Management
  * Git, Docker, SQL, Linux
* Machine Learning and Data Analysis
  * Deep learning frameworks: TensorFlow, PyTorch, Keras
  * Data manipulation tools: pandas, NumPy
  * 


Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
