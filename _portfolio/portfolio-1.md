---
title: "Evolution Controller for Foraging Behaviour"
excerpt: "Optimize the foraging behavior of robotic controllers through reinforcement learning and evolutionary algorithms.<br/><img src='/images/P11.png'>"
collection: portfolio
---

This project investigates the development of a robot capable of adapting to a dynamic and challenging simulated environment using principles derived from evolutionary algorithms. The robot, inspired by Braitenberg's vehicles, is designed to navigate a simulated world where it must avoid traps and seek resources like food and water, which can unpredictably turn into deadly poisons. This setup tests the robot's ability to continuously adapt to changing conditions.

![P12](https://dashpulsar.github.io/images/P12.png)

The method involves creating a robot with simple sensorimotor links, equipped with two batteries and sensors that respond to different simulated "light sources" representing food, water, and poison. The robot's actions are governed by the outputs of these sensors processed through a genetically encoded activation function. This function evolves across generations to improve the robot’s performance in foraging and avoiding hazards.

The robot operates on a 10x10 map where the location of resources and hazards is randomized. Its survival hinges on efficiently gathering food and water while dodging poisons. The evolutionary process optimizes the sensor activation functions using a genetic algorithm that considers the robot's battery life and foraging efficiency.

![P13](https://dashpulsar.github.io/images/P13.png)

Results show that the evolved robot can adeptly navigate its environment, making strategic decisions based on resource availability and battery status. It exhibits behaviors like preference for closer and safer resources and a shift in resource gathering strategy based on battery levels.

The discussion highlights the robot's proficiency in foraging and hazard avoidance and suggests potential future enhancements such as dynamic control systems and competitive scenarios with multiple robots. This project illustrates the potential of evolutionary algorithms in developing autonomous robots that can adapt to complex and unpredictable environments.
