.. Copyright 2020 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _undirected_adjacency_vector_graph:

=================================
Undirected adjacency vector graph
=================================

Class ``undirected_adjacency_vector_graph`` is the implementation of
:capterm:`undirected <Undirected graph>` :capterm:`weighted <Weighted graph>`
sparse graph concept with :capterm:`adjacency matrix` underneath for which the following
is true:

- The data within the graph is sparse and stored in :capterm:`CSR <CSR data>`.
- The :ref:`specific graph traits <api_graph_traits>` are defined for this class.

---------------------
Programming interface
---------------------

Refer to :ref:`API Reference: Undirected Adjacency Vector Graph <api_undirected_adjacency_vector_graph>`.