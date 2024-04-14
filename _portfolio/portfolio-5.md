---
title: "Optimization of Tunnel Excavation Robots Using Reinforcement Learning"
excerpt: "<br/><img src='/images/P51.png'>"
collection: portfolio
---

This is a non-open source project.

As a previous intern at HyperTunnel, I had the unique opportunity to contribute to a pioneering project aimed at optimizing the efficiency of tunnel excavation robots. These robots play a critical role in modern infrastructure projects, facilitating the construction of tunnels by creating multiple drilling points in pre-installed pipes. This project focused on enhancing the traditional genetic algorithm approach with a more advanced reinforcement learning technique, specifically Q-Learning, to improve the decision-making process for robotic actions.


In typical operations, the excavation process begins by installing several parallel pipelines, such as nine 100-meter pipes with a diameter of 40 centimeters each, throughout the tunnel. Robots are then dispatched to drill 20 holes per pipeline. Each drilling site typically requires about 20 hours of excavation. However, the efficiency of excavation can be significantly improved when adjacent pipelines work in concert to excavate the same area, thereby doubling the operational speed.

The objective of this project was to optimize the path and operation strategy for these robots to enhance efficiency, reduce time, and minimize operational costs in various tunnel environments that differ in the number of pipelines and drilling sites.


The primary challenge in this project was the complexity of decision-making for the robotic operations, given the variable number of holes and pipelines, as well as the unstructured nature of the underground environment. Initially, the project utilized a genetic algorithm to determine optimal drilling sequences and pathways for the robots. This method, which was not developed by me but was the foundation for our project's approach, involved simulating different strategies and selecting the best performer through a process mimicking natural selection.


Building on the existing use of genetic algorithms, I proposed the integration of a Q-Learning model, a form of reinforcement learning. This approach allowed the robots to learn from their environment interactively and adjust their strategies based on real-time feedback, rather than solely relying on pre-programmed simulations.

Q-Learning works by creating a "Q-table" or matrix that represents the "quality" of taking an action in a given state. As robots operate, they update the Q-table with values reflecting the success of different actions, thereby learning the most effective strategies for excavation. This method proved to be more flexible and dynamic compared to the rigid genetic algorithm, as it adapted continuously to the changing environment of the tunnel.


Implementing Q-Learning involved developing a simulation environment where different operational strategies could be tested dynamically. The robots were equipped with sensors and feedback mechanisms that allowed them to record the outcomes of various actions, continuously improving their decision-making processes through trial and error. The results were highly encouraging. The Q-Learning approach led to a 30% reduction in excavation time on average compared to the genetic algorithm. Moreover, it enhanced the robots' ability to deal with unexpected geological features and obstacles, thereby reducing downtime and mechanical failures.


The integration of Q-Learning into the HyperTunnel project not only optimized the operational efficiency of tunnel excavation robots but also demonstrated the potential for advanced machine learning techniques in real-world engineering applications. This project stands as a testament to the power of combining innovative thinking with cutting-edge technology to solve complex problems in infrastructure development.

This experience has profoundly shaped my understanding of both the capabilities and potential of automation in large-scale engineering projects. It underscores the importance of interdisciplinary approaches and continuous innovation in the field of robotic engineering.