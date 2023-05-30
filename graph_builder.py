import graphviz as graphviz

graph = graphviz.Digraph()

graph.edge('7 AM, start', 'check mails')
graph.edge('check mails', 'practice DSA, till 10 AM')
graph.edge('practice DSA, till 10 AM', 'practice SQL, till 2 PM')
graph.edge('practice SQL, till 2 PM', 'practice Spark & Cloud, till 7 PM')
graph.edge('practice Spark & Cloud, till 7 PM', 'work on projects around Data Engineering', label='often')
graph.edge('work on projects around Data Engineering', 'end')
graph.edge('practice Spark & Cloud, till 7 PM', 'work on projects around Data Analysis', label='sometimes')
graph.edge('work on projects around Data Analysis', 'end')