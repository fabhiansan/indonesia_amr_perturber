import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Your existing code
from data_perturber.insertion import entity_error_insertion
import penman

amr_string2 = """
(z0 / hancur-01
    :ARG0 (z1 / orang
              :mod (z2 / negara
                       :wiki "Israel"
                       :name (z3 / nama
                                 :op1 "Israel"))
              :ARG0-of (z4 / punya-peran-org-91
                           :ARG1 (z5 / keamanan)))
    :ARG1 (z6 / rumah
              :poss (z7 / orang
                        :quant 3
                        :mod (z8 / negara
                                 :wiki "Palestina"
                                 :name (z9 / nama
                                           :op1 "Palestina"))
                        :ARG1-of (z10 / tewas-01))))
"""

print("--- Running entity_error_insertion ---")
perturbed, changelog = entity_error_insertion(amr_string2)
print("--- Finished entity_error_insertion ---")


print("\nPerturbed AMR:")
try:
    print(penman.encode(perturbed))
except Exception as e:
    print(f"Error encoding perturbed graph: {e}")
    print("Original graph object:", perturbed)

print("\nChangelog:")
print(changelog)
