:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_json)).
:- use_module(library(http/http_parameters)).


server_start(Port) :- http_server(http_dispatch, [port(Port)]).
server_stop(Port) :- http_stop_server(Port, []).


:- http_handler('/api/hypgen', handle_hypgen, []).
:- http_handler('/api/hypgen/candidate_genes', handle_candidate_genes, []).
:- http_handler('/api/query', handle_query, []).

:- http_handler('/api/error', 
    throw(http_reply(server_error(json{message:'Internal server error'}))), 
    []).

handle_hypgen(Request) :-
    http_parameters(Request, 
              [rsid(RsId, [optional(false)]), 
               samples(Samples, [integer, optional(false)])]),

        Snp = snp(RsId),
        ((proof_tree(relevant_gene(Gene, Snp), Samples, Graph)) -> 
            reply_json(json{
                rsid: RsId,
                response: Graph
            })
        ;   reply_json(json{
                rsid: RsId,
                response: []
            })).

handle_candidate_genes(Request) :-
    http_parameters(Request, 
            [rsid(RsId, [optional(false)])]),

    Snp = snp(RsId),
    (candidate_genes(Snp, Genes) -> 
        reply_json(json{
            rsid: RsId,
            candidate_genes: Genes
        })
    ;   reply_json(json{
            rsid: RsId,
            candidate_genes: []
        })
    ).

handle_query(Request) :-
    http_parameters(Request, 
            [query(QueryString, [optional(false)])]),
    
    catch(
        (
            term_string(Query, QueryString),
            % Handle different query patterns
            (   Query = gene_id(Name, X) ->
                findall(X, gene_id(Name, X), Results)
            ;   Query = gene_name(Gene, X) ->
                findall(X, gene_name(Gene, X), Results)
            ;   Query = variant_id(Variant, X) ->
                findall(X, variant_id(Variant, X), Results)
            ;   Query = term_name(Term, Name) ->
                findall(Term, term_name(Term, Name), Results)
            ;   Query = maplist(Pred, List, X) ->
                (call(Query) -> 
                    Results = X
                ;   Results = []
                )
            ;   % Generic fallback - try to find X variable in any position
                (functor(Query, _, Arity),
                 Arity > 0 ->
                    findall(Query, call(Query), Results)
                ;   Results = []
                )
            ),
            reply_json(Results)
        ),
        Error,
        (
            format(string(ErrorMsg), "Query execution failed: ~w", [Error]),
            reply_json(json{error: ErrorMsg})
        )
    ).