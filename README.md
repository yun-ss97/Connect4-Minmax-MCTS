# Connect4-Minmax-MCTS
https://youtu.be/2WCTQIfpiLA

### Operating Environment

- programming language : Python 3.7 
- OS : Windows 10 
- Requirements : numpy ==1.16.3 
- Editor: JetBrains Pycharm Community Edition 2018.3.2 x64

### PC Environment

- version: Windows 10 Home 
- processor: i7-9700K CPU 
- Memory(RAM): 16.0GB 
- System Type: 64 bit

### Connect4 전략 설명

창의적인 전략을 도출하기 위해 Minmax 알고리즘에서 새로운 휴리스틱 함수를 정의하였고, MCTS 에서도 여러 논문을 참고하여 가장 좋은 성능을 보이는 평가함수를 적용하였다. 또한, 성능 평가를 통해 2가지 전략 모두에서 최고의 성능을 보이는 parameter 값을 찾아냈다.
더 나아가, 선공/후공 여부와 상대방이 사용하는 전략 등의 변수에 무관하게 일관적 성능을 보일 수 있도록 좋은 성능을 보이는 2개의 알고리즘을 랜덤 선택하게
했다. 성능 평가를 거쳐 최종 MCTS과 Minmax 알고리즘의 선택 비율은 8:2 로 정하였다.
