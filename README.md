# deep_learning

### what is the difference between ML and DL?

机器学习，就是利用计算机、概率论、统计学等知识，通过给计算机程序输入数据，让计算机学会新知识，是实现人工智能的途径，但这种学习不会让机器产生意识。机器学习的过程，就是通过训练数据寻找目标函数。数据质量会影响机器学习精度，所以数据预处理非常重要。

深度学习是机器学习的一种，现在深度学习比较火爆。在传统机器学习中，手工设计特征对学习效果很重要，但是特征工程非常繁琐。而深度学习能够从大数据中自动学习特征，这也是深度学习在大数据时代受欢迎的一大原因。是机器学习的一类，本质上就是之前机器学习的神经网络算法。

深度学习目前在语音识别、图像识别等领域特别在行。

### what is the difference between keras and tensorflow?

......(体会不深因keras用的太少)

### what is the difference between numpy and pandas?

numpy: calculation on data like matrix multiply, etc.
pandas: load and format data

### largest data amount you handled

see BMA-2.docx Bayer Project

### what is reinforcement learning?

人的一生其实都是不断在强化学习，当你有个动作（action）在某个状态（state）执行，然后你得到反馈（reward），尝试各种状态下各种动作无数次后，这几点构成脑中的马尔可夫模型，使你知道之后的行为什么为最优。所以你现在才知道什么东西好吃，什么东西好玩。

加强学习最重要的几个概念：agent，环境，reward，policy，action。环境通常利用马尔可夫过程来描述，agent通过采取某种policy来产生action，和环境交互，产生一个reward。之后agent根据reward来调整优化当前的policy。

例子：撩妹的过程就是一个优化问题。你的每一时刻的行为会对你最终撩妹是否成功，以多大的收益成功都会有影响。那么，你就会考虑，每一步采取什么行为才能（最优）撩妹！这可以看作一个RL问题。你肯定迫不及待的想知道怎么去求解了！假设1:你是第一次撩妹。那么你会去求教他人，逛各种论坛，总之收集大量相关知识。这个过程就是experience data。利用离线数据来train一个model。假设2:过去你有很多撩妹经验。你似乎又发现总是按照套路来并不能成功。嗯，经典的探索与利用问题，于是你尝试了其他方法，你发现获得了更好的效果。嗯，more optimal policy将上述过程对应到RL中：action：你的行为state：你观察到的妹子的状态reward：妹子的反应：开心or不开心。

出处： https://www.zhihu.com/question/31140846