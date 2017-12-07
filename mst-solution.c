#include <stdlib.h>
#include <assert.h>

#define _DEBUG

/**
** UTILITY
** Various utilities for the algorithms
**/

typedef struct Edge {
    int i;
    int j;
    int w;
} Edge;

void init_edge(Edge* edge, int i, int j, int w) {
    edge->i = i < j ? i : j;
    edge->j = j > i ? j : i;
    edge->w = w;
}

int create_edges(int N, int* adj, Edge* edges) {
    int numEdge = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            if (adj[i*N+j] != 0) {
                init_edge(&edges[numEdge], i, j, adj[i*N+j]);
                numEdge += 1;
            }
        }
    }
    return numEdge;
}

int cmp_edges(const void* left_ptr, const void* right_ptr) {
    const Edge* a = (const Edge*)left_ptr;
    const Edge* b = (const Edge*)right_ptr;
    if (a->w != b->w)
        return a->w - b->w;
    if (a->i != b->i)
        return a->i - b->i;
    return a->j - b->j;
}

void print_tree(Edge* tree, int nbEdges) {
    int sum = 0;
    for (int edge = 0; edge < nbEdges; ++edge) {
        printf("%d %d\n", tree[edge].i, tree[edge].j);
        sum += tree[edge].w;
    }
    #ifdef DEBUG
    printf("Sum : %d\n", s);
    #endif
}

/**
** Declarations of main procedures
**/

void sequential_prim(int N, int M, int *adj, Edge *tree);
int sequential_kruskal(int N, int M, int *adj, Edge *tree);
void parallel_prim(int procRank, int numProcs, int *adj, int N, Edge *tree);
void parallel_kruskal(int procRank, int numProcs, int *adj, int N, int M, Edge *tree);

/** Computing the Minimum Spanning Tree of a graph
 * @param N the number of vertices in the graph
 * @param M the number of edges in the graph
 * @param adj the adjacency matrix
 * @param algoName the name of the algorithm to be executed
 */
void computeMST(
    int N,
    int M,
    int *adj,
    char *algoName)
{
    int procRank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    Edge* tree = calloc(N-1, sizeof(Edge));

    if (strcmp(algoName, "prim-seq") == 0) { // Sequential Prim's algorithm
        if (procRank == 0) {
            if (numProcs != 1) {
                printf("ERROR: Sequential algorithm is ran with %d MPI processes.\n", numProcs);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        sequential_prim(N, M, adj, tree);
    } else if (strcmp(algoName, "kruskal-seq") == 0) { // Sequential Kruskal's algorithm
        if (procRank == 0) {
            if (numProcs != 1) {
                printf("ERROR: Sequential algorithm is ran with %d MPI processes.\n", numProcs);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        sequential_kruskal(N, M, adj, tree);
    } else if (strcmp(algoName, "prim-par") == 0) { // Parallel Prim's algorithm
        parallel_prim(procRank, numProcs, adj, N, tree);
    } else if (strcmp(algoName, "kruskal-par") == 0) { // Parallel Kruskal's algorithm
        parallel_kruskal(procRank, numProcs, adj, N, M, tree);
    } else { // Invalid algorithm name
        if (procRank == 0) {
            printf("ERROR: Invalid algorithm name: %s.\n", algoName);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (procRank == 0) {
        print_tree(tree, N-1);
    }
}

/**
** IMPLEMENTATION OF SEQUENTIAL KRUSKAL
** Using Path compression and Union by rank
** Optimal complexity
**/

typedef struct Node {
    int father;
    int rank;
} Node;

Node* create_nodes(int N) {
    Node* nodes = (Node*)calloc(N, sizeof(Node));
    for (int node = 0; node < N; ++node) {
        nodes[node].father = node;
        nodes[node].rank = 1;
    }
    return nodes;
}

int find(Node* nodes, int node) {
    if (nodes[node].father != node) {
        nodes[node].father = find(nodes, nodes[node].father);
    }
    return nodes[node].father;
}

void fusion(Node* nodes, int root1, int root2) {
    if (nodes[root1].rank > nodes[root2].rank) {
        fusion(nodes, root2, root1);
        return ;
    }
    nodes[root1].father = root2;
    if (nodes[root1].rank == nodes[root2].rank) {
        nodes[root2].rank += 1;
    }
}

int union_find(Edge* edges, int N, int M, Edge *tree) {
    Node* nodes = create_nodes(N);
    int numEdge = 0;
    for (int edge = 0; edge < M && numEdge < N-1; ++edge) {
        int root1 = find(nodes, edges[edge].i);
        int root2 = find(nodes, edges[edge].j);
        if (root1 != root2) {
            fusion(nodes, root1, root2);
            memcpy(tree+numEdge, edges+edge, sizeof(Edge));
            numEdge += 1;
        }
    }
    free(nodes);
    return numEdge;
}

int sequential_kruskal(int N, int M, int *adj, Edge* tree) {
    Edge* edges = calloc(M, sizeof(Edge));
    create_edges(N, adj, edges);
    qsort(edges, M, sizeof(Edge), cmp_edges);
    int nbEdges = union_find(edges, N, M, tree);
    free(edges);
    return nbEdges;
}

/**
** IMPLEMENTATION OF PARALLEL KRUSKAL
** Using Point to Point communications
**/

int create_forest(int procRank, int *adj, int nbRows, int N, Edge* forest) {
    Edge* edges = calloc(nbRows*nbRows, sizeof(Edge));
    int nbEdges = create_edges(nbRows, adj, edges);
    qsort(edges, nbEdges, sizeof(Edge), cmp_edges);
    int forestSize = union_find(edges, nbRows, nbEdges, forest);
    free(edges);
    return forestSize;
}

int receive_edges_from(int stepSize, int targetRank, Edge* edges) {
    int nbEdges;
    MPI_Recv(&nbEdges, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int* buffer = calloc(nbEdges * 3, sizeof(int));
    MPI_Recv(buffer, nbEdges * 3, MPI_INT, targetRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < nbEdges; ++i) {
        int* edge = &buffer[i*3];
        edges[i].i = edge[0];
        edges[i].j = edge[1];
        edges[i].w = edge[2];
    }
    return nbEdges;
}

void merge_sorted_lists(Edge* li, int ni, Edge* lj, int nj, Edge* output) {
    int i = 0;
    int j = 0;
    while (i < ni && j < nj) {
        if (j >= nj || (i < ni && cmp_edges(li+i, lj+j) < 0)) {
            memcpy(output+i+j, li+i, sizeof(Edge));
            i += 1;
        } else {
            memcpy(output+i+j, lj+j, sizeof(Edge));
            j += 1;
        }
    }
}

void lazy_merge(Edge** oldBuf, int n1, Edge** newBuf, int n2) {
    Edge* output = calloc(n1+n2, sizeof(Edge));
    merge_sorted_lists(*oldBuf, n1, *newBuf, n2, output);
    free(*oldBuf);
    *oldBuf = output;
}

int receive_edges(int procRank, int stepSize, int squareSize, Edge* edges, int nbEdges, int nbEdgesMax) {
    Edge* tmpBuf = calloc(nbEdgesMax, sizeof(Edge));
    Edge* buf1 = edges;
    Edge* buf2 = tmpBuf;
    Edge* newEdges = calloc(stepSize * squareSize, sizeof(Edge));
    for (int i = 0; i < stepSize; ++i) {
        int nbNewEdges = receive_edges_from(stepSize, procRank + stepSize + i, newEdges);
        merge_sorted_lists(buf1, nbEdges, newEdges, nbNewEdges, buf2);
        nbEdges += nbNewEdges;
        Edge* tmp = buf1;
        buf1 = buf2;
        buf2 = tmp;
    }
    if (stepSize & 1) {
        memcpy(edges, tmpBuf, nbEdges*sizeof(Edge));
    }
    free(newEdges);
    free(tmpBuf);
    return nbEdges;
}

int receive_forest(int procRank, int stepSize, Edge* edges) {
    int target = procRank - stepSize;
    return receive_edges_from(stepSize, target, edges);
}

int receive_new_edges(
    int procRank, int stepSize, int squareSize, int nbRows,
    Edge* edges, int nbMaxEdges, Edge* forest, int forestSize) {
    int newForestSize = receive_forest(procRank, stepSize, edges);
    printf("coucou\n");fflush(stdout);
    int nbNewEdges = receive_edges(procRank, stepSize, squareSize, edges, newForestSize, nbMaxEdges);
    lazy_merge(&edges, nbNewEdges, &forest, forestSize);
    return nbNewEdges;
}

void send_edges_to(int target, Edge* edges, int nbEdges) {
     int* buf = calloc(nbEdges*3, sizeof(int));
     for (int i = 0; i < nbEdges; ++i) {
         int* edge = buf+i*3;
         edge[0] = edges[i].i;
         edge[1] = edges[i].j;
         edge[2] = edges[i].w;
     }
     MPI_Send(&nbEdges, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
     MPI_Send(buf, nbEdges * 3, MPI_INT, target, 0, MPI_COMM_WORLD);
     free(buf);
}

int add_edges_from_submatrix(int procRank, int stepSize, int nbRows, int* adj, int N, Edge* edges) {
    int start = (procRank/stepSize - stepSize)*nbRows;
    int numEdge = 0;
    for (int i = 0; i < nbRows; ++i) {
        for (int j = start; j < start+nbRows*stepSize; ++j) {
            if (adj[i*N + j] == 0)
                continue ;
            init_edge(edges+numEdge, i, j, adj[i*N + j]);
            numEdge += 1;
        }
    }
    return numEdge;
}

void send_bipartite_forest(int procRank, int stepSize, int squareSize, int nbRows, int* adj, int N) {
    int target = procRank - stepSize;
    Edge* edges = calloc(squareSize*stepSize*stepSize, sizeof(Edge));
    int nbEdges = add_edges_from_submatrix(procRank, stepSize, nbRows, adj, N, edges);
    qsort(edges, nbEdges, sizeof(Edge), cmp_edges);
    Edge* forest = calloc(stepSize*nbRows-1, sizeof(Edge));
    int forestSize = union_find(edges, N, nbEdges, forest);
    send_edges_to(target, forest, forestSize);
    free(forest);
    free(edges);
}

void send_forest(int procRank, int stepSize, Edge* forest, int forestSize) {
    int target = procRank - stepSize;
    send_edges_to(target, forest, forestSize);
}

// works only if p | N and p = 2^k
void parallel_kruskal(int procRank, int numProcs, int *adj, int N, int M, Edge* tree) {
    int nbRows = N / numProcs;
    int squareSize = nbRows * nbRows;
    Edge* forest = calloc(nbRows-1, sizeof(Edge));
    int forestSize = create_forest(procRank, adj, nbRows, N, forest);
    for (int stepSize = 1, rank = procRank; stepSize*nbRows < N; stepSize <<= 1) {
        printf("MSG : %d %d %d\n", procRank, forestSize, stepSize);
        fflush(stdout);
        if (rank & 1) {
            send_bipartite_forest(procRank, stepSize, squareSize, nbRows, adj, N);
            if (procRank % stepSize == 0) {
                send_forest(procRank, stepSize, forest, forestSize);
            }
        } else {
            int subNbEdges = stepSize*(nbRows*(stepSize+1)-1)  + 2*(stepSize*nbRows-1);
            int nbMaxEdges = (M < subNbEdges) ? M : subNbEdges;
            Edge* edges = calloc(nbMaxEdges, sizeof(Edge));
            printf("lolilol1\n");fflush(stdout);
            int nbNewEdges = receive_new_edges(procRank, stepSize, squareSize, nbRows, edges, nbMaxEdges, forest, forestSize);
            printf("lolilol2\n");fflush(stdout);
            forest = realloc(forest, (stepSize*nbRows*2 - 1) * sizeof(Edge));
            printf("N: %d\n", nbNewEdges);
            forestSize = union_find(edges, N, nbNewEdges, forest);
            free(edges);
            rank >>= 1;
        }
        printf("MSG : %d %d %d\n", procRank, forestSize, stepSize);
        fflush(stdout);
    }
    printf("MSG : %d %d\n", procRank, forestSize);
    fflush(stdout);
    if (procRank == 0) {
        assert(forestSize == N-1);
        memcpy(tree, forest, sizeof(Edge) * forestSize);
    }
    free(forest);
}

/**
** IMPLEMENTATION OF SEQUENTIAL PRIM
** Using a heap
** Optimal complexity
**/

typedef struct Heap {
    int sizeMax;
    int curEnd;
    Edge* data;
} Heap;

Heap* create_heap(int nbMaxEdges) {
    Heap* heap = malloc(sizeof(heap));
    heap->sizeMax = nbMaxEdges;
    heap->curEnd = 1;
    heap->data = calloc(nbMaxEdges+1, sizeof(Edge));
    return heap;
}

void destroy_heap(Heap* heap) {
    free(heap->data);
    free(heap);
}

inline int father(int i) {return i/2;}
inline int left(int i) {return 2*i;}
inline int right(int i) {return 2*i+1;}

void swap_edges(Edge* a, Edge* b) {
    Edge tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

void add_edge(Heap* heap, Edge* edge) {
    memcpy(heap->data+heap->curEnd, edge, sizeof(Edge));
    int node = heap->curEnd;
    heap->curEnd += 1;
    while (
        father(node) != 0 &&
        cmp_edges(heap->data+father(node), heap->data+node) > 0
    ) {
        swap_edges(heap->data+father(node), heap->data+node);
        node = father(node);
    }
}

Edge* extract_min(Heap* heap) {
    Edge* top = malloc(sizeof(Edge));
    memcpy(top, heap->data+1, sizeof(Edge));
    heap->curEnd -= 1;
    swap_edges(heap->data+1, heap->data+heap->curEnd);
    int node = 1;
    while (
        (left(node)  <= heap->curEnd && cmp_edges(heap->data+node, heap->data+left(node))  > 0 ) ||
        (right(node) <= heap->curEnd && cmp_edges(heap->data+node, heap->data+right(node)) > 0)
    ) {
        int next = (cmp_edges(heap->data+left(node), heap->data+right(node)) < 0) ? left(node) : right(node);
        swap_edges(heap->data+node, heap->data+next);
        node = next;
    }
    return top;
}

void add_neighbors(int node, int N, int *adj, int* isVisited, Heap* heap) {
    for (int neighbor = 0; neighbor < N; ++neighbor) {
        if (adj[node*N + neighbor] != 0 && isVisited[neighbor] == 0) {
            Edge edge;
            init_edge(&edge, node, neighbor, adj[node*N + neighbor]);
            add_edge(heap, &edge);
        }
    }
}

void sequential_prim(int N, int M, int *adj, Edge* edges) {
    int curEdge = 0;
    int* isVisited = calloc(N, sizeof(int));
    Heap* heap = create_heap(M);
    isVisited[0] = 1;
    add_neighbors(0, N, adj, isVisited, heap);
    while (heap->curEnd != 1) {
        Edge* edge = extract_min(heap);
        int node = (!isVisited[edge->i])? edge->i : edge->j;
        if (isVisited[node])
            continue ;
        memcpy(edges+curEdge, edge, sizeof(Edge));
        curEdge += 1;
        isVisited[node] = 1;
        add_neighbors(node, N, adj, isVisited, heap);
    }
    qsort(edges, N-1, sizeof(Edge), cmp_edges);
    destroy_heap(heap);
    free(isVisited);
}

/**
** IMPLEMENTATION OF PARALLEL PRIM
** Using All to One communications
**/

typedef struct BorderNode {
    int w;
    int z;
} BorderNode;

BorderNode* create_border(int nbRows, int* isVisited, int N, int *adj) {
    BorderNode* border = calloc(nbRows, sizeof(BorderNode));
    int firstNode = 0;
    isVisited[firstNode] = 1;
    for (int y = 0; y < nbRows; ++y) {
        border[y].w = adj[y*N + firstNode];
        border[y].z = firstNode;
    }
    return border;
}

int find_closest_border(int procRank, BorderNode* border, int* isVisited, int nbRows) {
    int yOpt = -1;
    Edge smallest;
    #ifdef DEBUG
    printf("P%d ", procRank);
    #endif
    for (int y = 0; y < nbRows; ++y) {
        if (isVisited[y + procRank*nbRows]) {
            #ifdef DEBUG
            printf("# ");
            #endif
            continue ;
        }
        #ifdef DEBUG
        printf("%d ", border[y].w);
        #endif
        Edge curEdge;
        init_edge(&curEdge, procRank*nbRows + y, border[y].z, border[y].w);
        if (border[y].w != 0 &&
            (yOpt == -1 || cmp_edges((void*)&smallest, (void*)&curEdge) > 0)) {
            yOpt = y;
            memcpy(&smallest, &curEdge, sizeof(Edge));
        }
    }
    #ifdef DEBUG
    printf(" => %d ~ %d\n", yOpt, yOpt + procRank*nbRows);
    #endif
    return yOpt;
}

int* send_edge(int procRank, int numProcs, BorderNode* border, int nbRows, int y) {
    int edge[3];
    edge[0] = y;
    if (y != -1) {
        edge[0] = y + procRank*nbRows;
        edge[1] = border[y].z;
        edge[2] = border[y].w;
        #ifdef DEBUG
        printf("P%d send (%d %d %d)\n", procRank, edge[0], edge[1], edge[2]);
        #endif
    }
    int* minEdges = NULL;
    if (procRank == 0)
        minEdges = calloc(numProcs*3, sizeof(int));
    MPI_Gather(edge, 3, MPI_INT, minEdges, 3, MPI_INT, 0, MPI_COMM_WORLD);
    return minEdges;
}

int select_new_vertex(int procRank, int numProcs, int* minEdges, Edge* smallest) {
    if (procRank != 0)
        return -1;
    int idSmallestVertex = -1;
    #ifdef DEBUG
    printf("S ");
    #endif
    for (int idProc = 0; idProc < numProcs; ++idProc) {
        int* edge = &minEdges[idProc * 3];
        if (edge[0] == -1) {
            #ifdef DEBUG
            printf("# ");
            #endif
            continue ;
        }
        #ifdef DEBUG
        printf("(%d %d %d) ", edge[0], edge[1], edge[2]);
        #endif
        Edge curEdge;
        init_edge(&curEdge, edge[0], edge[1], edge[2]);
        if (idSmallestVertex == -1 || cmp_edges((void*)smallest, (void*)&curEdge) > 0) {
            memcpy(smallest, &curEdge, sizeof(Edge));
            idSmallestVertex = edge[0];
        }
    }
    assert(idSmallestVertex != -1);
    #ifdef DEBUG
    printf("\n%d", idSmallestVertex);
    printf(" -> (%d %d %d)\n", smallest->i, smallest->j, smallest->w);
    #endif
    return idSmallestVertex;
}

void add_vertex_to_border(int procRank, int numProcs, int* adj, int N, int newVertex, BorderNode* border, int* isVisited, int nbRows) {
    #ifdef DEBUG
    printf("add %d\n", newVertex);
    #endif
    isVisited[newVertex] = 1;
    for (int y = 0; y < nbRows; ++y) {
        if (isVisited[procRank*nbRows + y])
            continue ;
        int w = adj[y*N + newVertex];
        if (w != 0 && (w < border[y].w || border[y].w == 0)) {
            border[y].w = w;
            border[y].z = newVertex;
        }
    }
}

void parallel_prim_iteration(int procRank, int numProcs, int *adj, int N, BorderNode* border, int* isVisited, int nbRows, Edge* newEdge) {
    int y_min = find_closest_border(procRank, border, isVisited, nbRows);
    int* minEdges = send_edge(procRank, numProcs, border, nbRows, y_min);
    int newVertex = select_new_vertex(procRank, numProcs, minEdges, newEdge);
    MPI_Bcast(&newVertex, 1, MPI_INT, 0, MPI_COMM_WORLD);
    add_vertex_to_border(procRank, numProcs, adj, N, newVertex, border, isVisited, nbRows);
    free(minEdges);
}

void parallel_prim(int procRank, int numProcs, int *adj, int N, Edge *tree) {
    int nbRows = ceil((float)N / (float)numProcs); // Be carefull
    int* isVisited = calloc(N, sizeof(int));
    BorderNode* border = create_border(nbRows, isVisited, N, adj);
    for (int edge = 0; edge < N-1; ++edge) {
        parallel_prim_iteration(procRank, numProcs, adj, N, border, isVisited, nbRows, tree + edge);
    }
    if (procRank == 0) {
        qsort(tree, N-1, sizeof(Edge), cmp_edges);
    }
    free(border);
    free(isVisited);
}
