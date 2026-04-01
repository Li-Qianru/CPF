# CPF
Personalized Forgetting Mechanism with Concept-Driven Knowledge Tracing (CPF)

This repository provides the official implementation of CPF, a Personalized Forgetting Mechanism with Concept-Driven Knowledge Tracing, proposed in the paper:

Personalized Forgetting Mechanism with Concept-Driven Knowledge Tracing  
Shanshan Wang, Ying Hu, Qianru Li, Xun Yang, Zhongzhou Zhang, Keyang Wang, Xingyi Zhang
Manuscript submitted to ACM

CPF is a knowledge tracing framework that explicitly models personalized forgetting behaviors and hierarchical relationships among knowledge concepts, aiming to more accurately simulate students’ learning and forgetting processes over time.

📌 Overview
Knowledge Tracing (KT) aims to model students’ evolving knowledge states based on their historical learning interactions. Existing forgetting-aware KT models mainly rely on time-based decay mechanisms and overlook:

Individual differences in cognitive abilities and forgetting rates

The influence of prerequisite relationships among knowledge concepts

To address these limitations, CPF introduces:

Personalized learning and forgetting mechanisms driven by student cognitive abilities

Concept-driven forgetting modeling using a directed prerequisite-successor concept matrix (P-matrix)

Forgetting–review dynamics to simulate long-term and short-term memory interactions

🧠 Key Features
Personalized Forgetting Modeling  
Learns student-specific learning gains and forgetting rates based on response accuracy, response time, and exercise difficulty.

Concept-Driven Forgetting Mechanism  
Models how forgetting a prerequisite concept affects related successor concepts via a directed concept graph.

Forgetting–Review Mechanism  
Dynamically adjusts forgetting gates by measuring similarity between adjacent knowledge states.

Interpretable Cognitive Modeling  
Explicitly separates discriminative (student-specific) and generalized (concept-structure) factors in forgetting.


📊 Datasets
CPF is evaluated on several public educational datasets, including:

ASSISTments

Ednet

Each dataset contains student–exercise interaction logs with timestamps, correctness labels, and concept annotations.
