%:- use_module(queries).
:- style_check(-discontiguous).
:- use_module(library(clpfd)).

:- dynamic hideme/1.
:- multifile hideme/1.

hideme([]).
hideme([Goal|Goals]) :-
  call(Goal),
  hideme(Goals).

regulatory_effect(S, G) :- 
    in_regulatory_region(S, Enh),
    associated_with(Enh, G).
    % tf_snp(Tf, S),
    % regulates(Tf, G),
    % binds_to(Tf, Tfbs),
    % overlaps_with(Tfbs, Enh), 
    % !.

coding_effect(S, G) :- 
  hideme([
    chr(S, Chr),
    start(S, Pos),
    ref(S, Ref),
    alt(S, Alt),
    has_coding_effect(G, Chr, Pos, Ref, Alt)
  ]).

in_regulatory_region(S, Enh) :-
    S = snp(_),
    (Enh = super_enhancer(E)
    ;Enh = enhancer(E)),
    within_k_distance(Enh, S, 50000, 1). %50,000kb obtained from dbsup

alters_tfbs(S, Tf, G) :-
    find_and_rank_tfs(S, Tf, G).
    % format('Alters TFBS: S: ~w, Tf: ~w, G: ~w~n', [S, Tf, G]).


load_enh_tfbs(Enh) :-
  chr(Enh, Chr),
  start(Enh, Start),
  end(Enh, End),
  load_tfbs_data(Chr, Start, End).

codes_for(G, P) :-
    transcribed_to(G, T),
    translates_to(T, P).

in_tad_with(S, G1) :- 
    (closest_gene(S, G1)
    ;
    (closest_gene(S, G2),
    in_tad_region(G2, T),
    in_tad_region(G1, T))).