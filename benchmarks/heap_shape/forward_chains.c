/*
 * Inspired by the linked-list traversal in SV-Benchmarks' heap-data/running_example.c.
 * The two allocated node pools have nondeterministic links.  Assumptions constrain
 * each link to a later node or to the end sentinel, making every chain acyclic.
 */
extern void *malloc(unsigned long size);
extern void free(void *ptr);

#define OUTER_NODES 32
#define INNER_NODES 8

typedef struct node {
    int next;
} node_t;

extern int nondet_int(void);

void reach_error(void) {
    __CPROVER_assert(0, "reach_error");
}

#define CONSTRAIN_LINK(pool, index, count) {                            \
    (pool)[index].next = nondet_int();                                  \
    __CPROVER_assume((pool)[index].next == -1 ||                         \
        ((pool)[index].next > (index) && (pool)[index].next < (count))); \
}

int main(void) {
    node_t *outer = malloc(OUTER_NODES * sizeof(*outer));
    node_t *inner = malloc(INNER_NODES * sizeof(*inner));
    __CPROVER_assume(outer != 0 && inner != 0);

    /* These constraints describe heap shape; they do not iterate over it. */
    CONSTRAIN_LINK(outer, 0, OUTER_NODES);
    CONSTRAIN_LINK(outer, 1, OUTER_NODES);
    CONSTRAIN_LINK(outer, 2, OUTER_NODES);
    CONSTRAIN_LINK(outer, 3, OUTER_NODES);
    CONSTRAIN_LINK(outer, 4, OUTER_NODES);
    CONSTRAIN_LINK(outer, 5, OUTER_NODES);
    CONSTRAIN_LINK(outer, 6, OUTER_NODES);
    CONSTRAIN_LINK(outer, 7, OUTER_NODES);
    CONSTRAIN_LINK(outer, 8, OUTER_NODES);
    CONSTRAIN_LINK(outer, 9, OUTER_NODES);
    CONSTRAIN_LINK(outer, 10, OUTER_NODES);
    CONSTRAIN_LINK(outer, 11, OUTER_NODES);
    CONSTRAIN_LINK(outer, 12, OUTER_NODES);
    CONSTRAIN_LINK(outer, 13, OUTER_NODES);
    CONSTRAIN_LINK(outer, 14, OUTER_NODES);
    CONSTRAIN_LINK(outer, 15, OUTER_NODES);
    CONSTRAIN_LINK(outer, 16, OUTER_NODES);
    CONSTRAIN_LINK(outer, 17, OUTER_NODES);
    CONSTRAIN_LINK(outer, 18, OUTER_NODES);
    CONSTRAIN_LINK(outer, 19, OUTER_NODES);
    CONSTRAIN_LINK(outer, 20, OUTER_NODES);
    CONSTRAIN_LINK(outer, 21, OUTER_NODES);
    CONSTRAIN_LINK(outer, 22, OUTER_NODES);
    CONSTRAIN_LINK(outer, 23, OUTER_NODES);
    CONSTRAIN_LINK(outer, 24, OUTER_NODES);
    CONSTRAIN_LINK(outer, 25, OUTER_NODES);
    CONSTRAIN_LINK(outer, 26, OUTER_NODES);
    CONSTRAIN_LINK(outer, 27, OUTER_NODES);
    CONSTRAIN_LINK(outer, 28, OUTER_NODES);
    CONSTRAIN_LINK(outer, 29, OUTER_NODES);
    CONSTRAIN_LINK(outer, 30, OUTER_NODES);
    CONSTRAIN_LINK(outer, 31, OUTER_NODES);

    CONSTRAIN_LINK(inner, 0, INNER_NODES);
    CONSTRAIN_LINK(inner, 1, INNER_NODES);
    CONSTRAIN_LINK(inner, 2, INNER_NODES);
    CONSTRAIN_LINK(inner, 3, INNER_NODES);
    CONSTRAIN_LINK(inner, 4, INNER_NODES);
    CONSTRAIN_LINK(inner, 5, INNER_NODES);
    CONSTRAIN_LINK(inner, 6, INNER_NODES);
    CONSTRAIN_LINK(inner, 7, INNER_NODES);

    unsigned outer_visits = 0;
    unsigned pair_visits = 0;
    int p = 0;
    while (p != -1) {
        ++outer_visits;
        int q = 0;
        while (q != -1) {
            ++pair_visits;
            q = inner[q].next;
        }
        p = outer[p].next;
    }

    if (outer_visits > OUTER_NODES ||
        pair_visits > OUTER_NODES * INNER_NODES)
        reach_error();

    free(inner);
    free(outer);
    return 0;
}
