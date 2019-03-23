from source.config import Config
from source.random_graph import RandomGraph

if __name__  == '__main__':
    config = Config().parse_args()
    graph = RandomGraph(config)
    graph.create_graph()
    print('created new random graph')