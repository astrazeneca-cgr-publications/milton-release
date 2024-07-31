'''Module implementing ICD10 node tree, which is a utility for working with the 
ICD10 code hierarchy (ontology). The tree is recursive composition of nodes each 
of which corresponding to an entity in the ICD10 ontology. The nodes provide
functionality for querying/listing their descendants, as well as means of
ICD10 code lookup/search. Features include:

* Convenient browsing of hierarchy with tab-completion (object attributes)
* Textual representation (all objects have printable string representations)
* Listing of all ICD10 codes in a subtree
* Listing of all subtrees k levels below the current one
* Mapping from raw ICD10 codes to their textual descriptions
'''

import re
import json
from functools import cached_property
from pathlib import Path


def extract_icd10_data_from_ukb(
    json_result_path=None,
    icd10_url='https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=41270'):
    """Utility for fetching ICD10 hierarchy data from the UK Biobank website and
    converting it to a form suitable for building of an ICD10Tree instance.
    The data is stored as JSON file and should be passed to ICD10Tree.load()
    method.
        
    Parameters
    ----------
    json_result_path : string, optional
      Destination path for the downloaded JSON data.
    icd10_url : Optional string
      Where to search for the data (if the default value no longer works).
    """
    
    # hiding imports here since this is a rarely used functionality
    from bs4 import BeautifulSoup
    import requests
    
    def find_leaves(tree):
        leaves = []
        for leaf in tree.find_all('li', class_=['tree_sub', 'tree_node'], recursive=False):
            label = leaf.find(['label', 'span'], class_='tree_desc', recursive=False)
            desc = label.text
            if label.name == 'label':
                # not a terminal node
                subtree = leaf.find('ul', class_='tree', recursive=False)
                leaves.append((desc, find_leaves(subtree)))
            else:
                leaves.append((desc, []))
        return leaves
    
    response = requests.get(icd10_url)
    if response.status_code == 200:
        page = BeautifulSoup(response.content, 'html.parser')
        data = find_leaves(page.find('ul', class_='tree'))
        
        if json_result_path is None:
            json_result_path = (Path(__file__).absolute().parent.parent 
                                / 'resources' 
                                / 'icd10.json')
            
        with open(json_result_path, 'w') as fout:
            json.dump(data, fout)
    else:
        raise Exception('Cannot fetch ICD10 hierarchy, got status code: '
                        + str(response.status_code))


class ICD10Tree:
    '''ICD10 phenotype hierarchy (a tree) - all ICD10 grouping  levels are 
    defined as object's attributes in a recursive manner. Use the __iter__ 
    method to list all concrete ICD10 codes from all sub-trees.
    Use the .subsections(n) method to list all sub-trees n-levels below the 
    current one in the hierarchy.    
    This class should not be istantiated with the constructor, but
    rather with the .load() static method which loads up a 
    JSON representation of the tree from a file and constructs the 
    recursive object structure.
    '''
    
    def __init__(self, subtree, full_text=None, is_leaf=False):
        """New ICD10 tree instance from ICD10 data processed 
        by extract_icd10_data_from_ukb() function.
        """
        self._parent = None
        self._contents = []
        self._full_text = full_text or 'ICD10 Hierarchy'
        self.full_name = self._extract_name(self._full_text)
        self.name = self.full_name.replace('.', '')
        self._codes = [self.name] if is_leaf else []
        self._descmap = {self.name: self._full_text}
        self._node_map = {}
        self._parse_tree(subtree)
        
    @classmethod
    def _extract_name(cls, description):
        """Extracts the name (ICD10 code/chapter name/section code) from
        description string
        """
        descr = cls._sanitize(description)
        if descr.startswith('Chapter'):
            return re.findall('Chapter [XIV]+', descr)[0]
        else:
            return descr.split(' ', maxsplit=1)[0]

    def _parse_tree(self, tree):
        def as_id(txt):
            return re.sub(r'[\s,\.-]+', '_', txt)
        
        for descr, subtree in tree:
            leaf_node = ICD10Tree(subtree, descr, not subtree)
            self._contents.append(leaf_node)
            attribute = as_id(leaf_node.name)
            setattr(self, attribute, leaf_node)
        
        for subtree in self._contents:
            subtree._parent = self
            self._codes.extend(subtree._codes)
            self._descmap.update(subtree._descmap)
            self._node_map[subtree.name] = subtree
            self._node_map.update(subtree._node_map)
            
        self._codes = sorted(self._codes)
        
    @cached_property
    def description(self):
        """Returns the textual description of the node (the full string
        representation of a node is: "<code> <description>")
        """
        return str(self)[len(self.full_name):].strip()
    
    @property
    def parent(self):
        return self._parent
        
    @staticmethod
    def _sanitize(text):
        text = re.sub('["\'`&\\*|]+', '', text)
        text = re.sub(r'[\\/]+', '-', text)
        return text
            
    def __getitem__(self, code):
        return self._descmap[code]
    
    def __contains__(self, code):
        """Returns True if the given ICD10 code have a corresponding node in
        the tree.
        """
        return code in self._descmap                
            
    def __iter__(self):
        """Returns an iterator over all ICD10 codes under this node.
        """
        return iter(self._codes)
            
    def __str__(self):
        """Returns a string representation of the node which contains its
        name and description.
        """
        return self._full_text
            
    def __repr__(self):
        """Returns this nodes textual description along with descriptions of
        all of its immediate descendants.
        """
        lines = [self._full_text] + ['  ' + str(c) for c in self._contents]
        return '\n'.join(lines)
    
    @cached_property
    def chapter(self):
        """Returns the chapter of this node or None if this is the root node.
        """
        if self.parent is None:
            return None

        node = self
        while node.parent.parent is not None:
            node = node.parent
        return node

    def subsections(self, level=1):
        """Returns a list of sub-nodes from given level below this node.
        """
        trees = []
        if level < 0:
            raise ValueError('Negative level requested.')
        if level == 0:
            return [self]
        for t in self._contents:
            trees.extend(t.subsections(level - 1))
        return trees
    
    def find_nodes(self, phrase, exact=False):
        """Finds all tree nodes that match the phrase in their description.
        """
        phrase = phrase.lower()
        nodes = []
        
        def search_description(tree):
            if phrase in str(tree).lower():
                nodes.append(tree)
                
        def exact_search(tree):
            if phrase == str(tree).lower():
                nodes.append(tree)
                
        search_fn = search_description if not exact else exact_search
        self._dfs(search_fn)
        return sorted(nodes, key=str)
    
    def find_by_code(self, code):
        """Finds a tree node with an exact matching of the code.
        """
        return self._node_map.get(code)
    
    def find_descr(self, code):
        """Searches for the description given the code. Returns None if the code 
        is not recognized.
        """
        root = self
        while root.parent is not None:
            root = root.parent
        return root._descmap.get(code)
    
    def find(self, phrase):
        """Returns a sorted list of code descriptions that contain 
        the given phrase. The search is *case-insensitive*.
        """
        phrase = phrase.lower()
        return sorted([d for d in list(self._descmap.values()) 
                       if phrase in d.lower()])
        
    def _dfs(self, visitor):
        """Depth-First Search - a utility.
        """
        finish = visitor(self)
        if not finish:
            for t in self._contents:
                finish = t._dfs(visitor)
                if finish:
                    break
        return finish
    
    @property
    def all_codes(self):
        """Returns unsorted list of all codes, including the "codes" of 
        higher-level constructs such as chapters.
        """
        return list(self._node_map)    
    
    @classmethod
    def load(cls, path=None):
        """Loads ICD10 data from a json file.
        
        Parameters
        ----------
        path : optional, string
          Path to the json file. When None, the default path is used
        """
        if path is None:
            pkg = Path(__file__).absolute().parent.parent
            path = pkg / 'resources' / 'icd10.json'
            
        with open(str(path), 'r') as f:
            return cls(json.load(f))
