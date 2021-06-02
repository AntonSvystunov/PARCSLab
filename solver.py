# Standard library imports
from math import sqrt
# Vector algebra

def vector_add(vec1, vec2):
    return [vec1[0]+vec2[0], vec1[1]+vec2[1]]

def vector_sub(vec1, vec2):
    return [vec1[0]-vec2[0], vec1[1]-vec2[1]]

def perpendicular(vec):
    return [-vec[1], vec[0]]

def list_equal(list1, list2):
    val1 = list1[0] == list2[0]
    val2 = list1[1] == list2[1]
    return val1 and val2

def list_divide(vec, val):
    """
    Returns
    -------
    out : list
        Input list 'vec' divided by the value 'val'.
    """
    return [vec[0]/val, vec[1]/val]

def normalise(vector, length=1):
    """
    Returns
    -------
    out : list
        Input list 'vec' normalsied to the value of 'length'.
    """
    norm = sqrt(vector[0]**2 + vector[1]**2)/length
    return list_divide(vector, norm)

# Sorting functions

def lexigraphic_sort(points):
    points_sorted = sorted(points, key=lambda k: [k[0], k[1]])
    return points_sorted

# Linear algebra

def in_circle(a, b, c, d):
    """
    This function is used to check whether a point 'd' is contained by the
    circle defined by three other points 'a', 'b', 'c'. This is achieved by
    calculating the sign of the following 4x4 matrix determinant.
    """
    c1 = a[0] - d[0]
    c2 = b[0] - d[0]
    c3 = c[0] - d[0]
    
    u1 = a[1] - d[1]
    u2 = b[1] - d[1]
    u3 = c[1] - d[1]

    v1 = c1**2 + u1**2
    v2 = c2**2 + u2**2
    v3 = c3**2 + u3**2
    
    det = c1*((u2*v3)-(v2*u3)) - c2*((u1*v3)-(v1*u3)) + c3*((u1*v2)-(v1*u2))
    
    return det < 0

def ccw_angle(p1, p2, p3):
    """
    Use the cross product to determine the acute angle defined by three
    points p1, p2, p3. Given this is a "right-handed" coordinate system there
    are three possible outcomes for the angle:
    1.  +ve angle, p3 lies to the right of the line defined by p1 p2
    2.  -ve angle, p3 lies to the left of the line defined by p1 p2
    3.  angle of 0, the points are collinear
    
    Parameters
    ----------
    p1, p2, p3 : list
        The three points being tested
    Returns
    -------
    angle : float
    """
    angle = (p1[0]-p3[0]) * (p2[1]-p3[1]) - (p1[1]-p3[1]) * (p2[0]-p3[0])
    return angle

def on_right(p1, p2, p3):
    """
    Returns
    -------
    out : Bool
        Return true if the point p3 is on the right of the line defined by the 
        points p1 and p3.
    """
    return ccw_angle(p1, p2, p3) > 0
    
def on_left(p1, p2, p3):
    """
    Returns
    -------
    out : Bool
        Return true if the point p3 is on the left of the line defined by the 
        points p1 and p3.
    """
    return ccw_angle(p1, p2, p3) < 0

"""
This module implements functions for splitting python lists in various ways.
"""

# Functions of points lists

def split_in_half(input_points):
    """
    This function takes in a list of points, splits this list in half and
    return the two new lists containing each subset of the input points.
    
    Parameters
    ----------
    input_points : list
        The set of points to be split
    Returns
    -------
    left : list
        The first half of the input points
    right : list
        The second half of the input points
    """
    mid_val = (len(input_points) + 1) // 2
    
    left = input_points[:mid_val]
    right = input_points[mid_val:]
    return left, right

def groups_of_2(points):
    num = len(points)
    if num%2==0:
        return [points[i:i+2] for i in range(0, num, 2)]
    elif num%2==1:
        split = [[points[0], points[1], points[3]]]
        split2 = [points[i:i+2] for i in range(3, num, 2)]
    return split + split2

def groups_of_3(points):
    num = len(points)
    if num%3==0:
        return [points[i:i+3] for i in range(0, num, 3)]
    elif num%3==1:
        split = [points[i:i+2] for i in range(0, 4, 2)]
        split2 = [points[i:i+3] for i in range(4, num, 3)]
    elif num%3==2:
        split = [[points[0], points[1]]]
        split2 = [points[i:i+3] for i in range(2, num, 3)]
    return split + split2

# Edge class  

class Edge():
    """
    This class represents a single edge as defined in the Guibas and Stolfi 
    edge algebra formalism.
    Attributes
    ----------
    index : int
        unique edge index
    org : int
        index of edge origin point
    dest : int
        index of edge destination point
    sym : int
        index of symetric edge
    onext : int
        index of next ccw edge connected to the origin point
    oprev : int
        index of previous ccw edge connected to the origin point
    deactivate : bool
        status of edge in triangulation. False if the edge is still part of
        the triangulation. 
    """
    def __init__(self, idx, org, dst, s, onxt, oprv):
        self.index = idx
        self.org = org
        self.dest = dst
        self.sym = s
        self.onext = onxt
        self.oprev = oprv
        self.deactivate = False
        
    def __repr__(self):
        return "[" + ",".join([self.index, self.org, self.dest, self.sym, self.onext, self.oprev, self.deactivate]) + "]"

    def return_point(self):
        return [self.org, self.dest]
    
    def shift_indices(self, shift_edges, shift_points):
        """
        This function is used to shift the indices of an edge. This is used
        when merging sets of edges to ensure each edge has a unique index. 
        Parameters
        ----------
        shift_edges : int
            integer to shift edge indices by
        shift_points : int
            integer to shift orgin and destination points by
        """
        self.index += shift_edges
        self.org += shift_points
        self.dest += shift_points
        self.sym += shift_edges
        self.onext += shift_edges
        self.oprev += shift_edges

    def find_connections(self, edges):
        """
        Find all the edges in the triangulation connected to the origin point
        of this edge. This gives a list of the points that the boid is to
        consider as neighbours. 
        Parameters
        ----------
        unique_edges : list of Edge class objects
            The list of unique edges in the triangulation
        Returns
        -------
        pts_subset : list
            List of the neighbour points
        """
        pts_subset = [self.return_point()]
        next_edge = edges[self.onext]
    
        while not list_equal(next_edge.return_point(), self.return_point()):
            pts_subset.append(next_edge.return_point())
            next_edge = edges[next_edge.onext]
        return pts_subset

def setup_edge(origin, dest, edge_idx):
    """
    This function takes in the index of two points and creates an edge array, 
    as well as an edge for the symetric edge.
    Parameters
    ----------
    origin : int
        index of origin point
    dest : int
        index of destination point
    edge_idx : int
        The index of the new edge
    Returns
    -------
    edge : Edge class
        Edge connecting org to dest
    edge_sym : Edge class
        Edge connecting dest to org
    """
    e1_idx = edge_idx
    e2_idx = edge_idx + 1

    edge = Edge(e1_idx, origin, dest, e2_idx, e1_idx, e1_idx)
    edge_sym = Edge(e2_idx, dest, origin, e1_idx, e2_idx, e2_idx)
    
    return edge, edge_sym

# Edges class 

class Edges():
    def __init__(self):
        self.edges = []
        self.num_edges = 0
        self.inner = None
        self.outer = None
        
    def push_back(self, new_edge):
        self.edges.append(new_edge)
        self.num_edges += 1
        
    def set_extreme_edges(self, left_most_edge, right_most_edge):
        self.inner = left_most_edge
        self.outer = right_most_edge
        
    def splice(self, edge1, edge2):
        """
        This function is used when adding another edge to a point in the 
        triangulation which has existing edges connected to it. The next and 
        previous ccw edges are updated accordingly for each of the input edges.
    
        Parameters
        ----------
        edge1 : Edge class
        edge2 : Edge class
        """
        self.edges[self.edges[edge1].onext].oprev = edge2
        self.edges[self.edges[edge2].onext].oprev = edge1

        onext_2 = self.edges[edge2].onext
        onext_1 = self.edges[edge1].onext
        
        self.edges[edge1].onext = onext_2
        self.edges[edge2].onext = onext_1
    
    def connect(self, edge1, edge2):
        """
        This function takes two seperated edges and creates a new edge 
        connecting the two.
    
        Parameters
        ----------
        edge1 : Edge class
        edge2 : Edge class
        Returns
        -------
        out : int
            index of the created edge
        """
    
        current_index = self.num_edges
        edge, edge_sym = setup_edge(self.edges[edge1].dest, self.edges[edge2].org, current_index)
        
        self.push_back(edge)
        self.push_back(edge_sym)
    
        edge1_sym_oprev = self.edges[self.edges[edge1].sym].oprev
        
        self.splice(edge.index, edge1_sym_oprev)
        self.splice(self.edges[edge.index].sym, edge2)
        
        return edge.index
        
    def kill_edge(self, e):
        """
        This function removes an edge from the triangulation by setting the 
        status of edge.deactivate to True. The function also fixed the
        connecting edges too.
    
        Parameters
        ----------
        e : Edge class
            edge to remove from the triangulation
        """
        # Fix the local triangulation 
        self.splice(e, self.edges[e].oprev)
        self.splice(self.edges[e].sym, self.edges[self.edges[e].sym].oprev)
        
        # Set the status of the edge and it's symetric edge to kill
        self.edges[e].deactivate = True
        self.edges[self.edges[e].sym].deactivate = True
        
    def filter_deactivated(self):
        self.edges = [e for e in self.edges if e.deactivate == False]

    def get_unique(self, num):
        unique = ['']*num
        points_seen = []
        for edge in self.edges:
            if edge.org not in points_seen:
                unique[edge.org] = edge
                points_seen.append(edge.org)
                
        return unique

# Triangulation class 

class TriangulationEdges(Edges):
    def __init__(self, points_subset):
        Edges.__init__(self)
        self.points = points_subset

    def shift_indices(self, shift_edges, shift_points):
        for edge in self.edges:
            edge.shift_indices(shift_edges, shift_points)
        
    def merge_hulls(self, second_hull):
        # Calculate the capacity of the new edges array
        len1 = self.num_edges
        len2 = second_hull.num_edges
        
        # Set the correct indices for the second hull
        second_hull.shift_indices(len1, len(self.points))

        # Combine the edges data from the two triangulations
        self.edges += second_hull.edges
        self.num_edges = len1 + len2
        
    def combine_triangulations(self, triangulation):
        self.merge_hulls(triangulation)
        self.points += triangulation.points
        return self

def line_primitive(pts_subset):
    """
    This function takes in a list of two points and forms an edge. 
    The symetric edge, where the origin and destination points are reversed, 
    is also constructed. These two edge are added into a new TriangulationEdges
    class object which is returned by the function. 
    
    Parameters
    ----------
    pts_index : list
        List of the indices of the two points
    pts_subset : lists of lists
        A set of two points with the form [ [x1, y1], [x2, y2] ]
    Returns
    -------
    left_most_edge : int
        Index of edge with the left most point
    right_most_edge : int
       Index of the edge with the right most point
    line : TriangulationEdges
        The resulting triangulation of two points
    """
    p1, p2 = 0, 1
    edge, edge_sym = setup_edge(p1, p2, 0)
    line = TriangulationEdges(pts_subset)
    line.push_back(edge)
    line.push_back(edge_sym)

    left_most_edge = edge.index
    right_most_edge = line.edges[edge.index].sym
    
    line.set_extreme_edges(left_most_edge, right_most_edge)
    return line

def triangle_primitive(pts_subset):
    """
    This function takes a list of three points and forms three edges to 
    create a single triangle. This triangle has the property that the origin
    of one edge is connected to the destination of the next edge in a CCW 
    orientation.
    Parameters
    ----------
    pts_index : list
        List of the indices of the three points
    pts_subset : lists of lists
        A set of three points with the form [ [x1, y1], [x2, y2] , [x3, y3] ]
    Returns
    -------
    out1 : int
        Index of edge with the left most point
    ou2 : int
       Index of the edge with the right most point
    edges : TriangulationEdges
        The resulting triangulation of three points
    """
    p1, p2, p3 = 0, 1, 2
    triang = TriangulationEdges(pts_subset)
    
    # Create the first two edges of the triangle
    edge1, edge1_sym = setup_edge(p1, p2, 0)
    triang.push_back(edge1)
    triang.push_back(edge1_sym)
    
    edge2, edge2_sym = setup_edge(p2, p3, 2)
    triang.push_back(edge2)
    triang.push_back(edge2_sym)
    
    triang.splice(edge1_sym.index, edge2.index)
    
    # To maintain the counter-clockwise orientation of the edges in the 
    # triangle, we determine where p3 is in relation to the two existing edges.
    pt1 = pts_subset[triang.edges[edge1.index].org]
    pt2 = pts_subset[triang.edges[edge1.index].dest]
    pt3 = pts_subset[p3]
    
    if on_right(pt1, pt2, pt3):
        # Points are in CCW orientiaton
        c = triang.connect(edge2.index, edge1.index)
        triang.set_extreme_edges(edge1.index, edge2_sym.index)
        return triang
    
    if on_left(pt1, pt2, pt3):
        # Points are in CW orientiaton
        c = triang.connect(edge2.index, edge1.index)
        triang.set_extreme_edges(triang.edges[c].sym, c)
        return triang
    
    # Points are collinear
    triang.set_extreme_edges(edge1.index, edge2_sym.index)
    return triang
    
def make_primitives(split_pts):
    primitives = []
    for pts_subset in split_pts:
        
        if len(pts_subset) == 2:
            # 2 points define a single edge
            primitives.append(line_primitive(pts_subset))
    
        elif len(pts_subset) == 3:
            # 3 points define a single triangle
            primitives.append(triangle_primitive(pts_subset))
    return primitives

# Edge finding functions 

def lowest_common_tangent(h_left, h_right):
    """
    Given two fully triangulated sets of points, this function finds an
    edge connecting the two triangulations. Each triangulation forms a convex 
    hull of edges. The edge to be found by this function is the edge with the 
    lowest y-value point which is still tangential to both hulls. This is
    known as the 'base' edge, as it is the first edge connecting two 
    separately triangulated point sets. 
    Parameters
    ----------
    h_left : TriangulationEdges
    h_right : TriangulationEdges
    Returns
    -------
    left_e : int
        The index of the edge in the right hull which forms one end of the 
        base edge
    right_e : int
        The index of the edge in the left hull which forms the other end of 
        the base edge
    """
    left_e = h_left.outer
    right_e = h_right.inner
    
    pts_left = h_left.points
    pts_right = h_right.points
    
    p1 = pts_left[h_left.edges[left_e].org]
    p2 = pts_left[h_left.edges[left_e].dest]

    p4 = pts_right[h_right.edges[right_e].org]
    p5 = pts_right[h_right.edges[right_e].dest] 

    while True:
        if on_right(p1, p2, pts_right[h_right.edges[right_e].org]):
            left_e = h_left.edges[h_left.edges[left_e].sym].onext

            p1 = pts_left[h_left.edges[left_e].org]
            p2 = pts_left[h_left.edges[left_e].dest]

        elif on_left(p4, p5, pts_left[h_left.edges[left_e].org]):
            right_e = h_right.edges[h_right.edges[right_e].sym].oprev
            p4 = pts_right[h_right.edges[right_e].org]
            p5 = pts_right[h_right.edges[right_e].dest]
            
        else:
            return left_e, right_e

def rcand_func(rhull, rcand, b1, b2):
    """
    This function finds the candidate edge from the right hull triangulation.
    An initial candidate 'rcand' is given. This candidate is tested. If the
    candidate fails it is deleted from the triangulation and the next 
    potential candiate is considered. While a valid candidate has not been 
    found this process continues until a valid candidate is found.
    Parameters
    ----------
    rhull : TriangulationEdges
        The triangulation of edges on the right hand side
    rcand : TYPE
        DESCRIPTION.
    b1 : list
        DESCRIPTION.
    b2 : list
        DESCRIPTION.
    Returns
    -------
    rhull : TriangulationEdges
        DESCRIPTION.
    rcand : TYPE
        DESCRIPTION.
    """
    completed = False
    while not completed:
        rcand_onext_dest = rhull.edges[rhull.edges[rcand].onext].dest
        rcand_dest = rhull.edges[rcand].dest
        ccw_test = on_right(b1, b2, rhull.points[rcand_onext_dest])
        next_cand_invalid = in_circle(b2, b1, 
                                             rhull.points[rcand_dest], 
                                             rhull.points[rcand_onext_dest])
        if ccw_test and next_cand_invalid:
            t = rhull.edges[rcand].onext
            rhull.kill_edge(rcand)
            rcand = t
        else:
            completed = True
    return rhull, rcand

def lcand_func(lhull, lcand, b1, b2):
    """
    This function performs the same task as the above 'rcand_func' but testing
    for the left candidate edge. 
    """
    completed = False
    while not completed:
        lcand_oprev_dest = lhull.edges[lhull.edges[lcand].oprev].dest
        lcand_dest = lhull.edges[lcand].dest
        ccw_test = on_right(b1, b2, lhull.points[lcand_oprev_dest])
        next_cand_invalid = in_circle(b2, b1, 
                                             lhull.points[lcand_dest], 
                                             lhull.points[lcand_oprev_dest])
        if ccw_test and next_cand_invalid:
            t = lhull.edges[lcand].oprev
            lhull.kill_edge(lcand)
            lcand = t
        else:
            completed = True
    return lhull, lcand

def candidate_decider(rcand, lcand, lcand_valid, triangulation):
    """
    Given two potential edges which could be added to the triangulation, 
    decide which of the edges is the correct one to add.
    Parameters
    ----------
    rcand : int
        index of right candidate edge
    lcand : int
        index of left candidate edge
    lcand_valid : TYPE
        DESCRIPTION.
    triangulation : TriangulationEdges
        DESCRIPTION.
    Returns
    -------
    result : bool
        DESCRIPTION.
    """
    pt1 = triangulation.points[triangulation.edges[rcand].dest]
    pt2 = triangulation.points[triangulation.edges[rcand].org]
    pt3 = triangulation.points[triangulation.edges[lcand].org]
    pt4 = triangulation.points[triangulation.edges[lcand].dest]
    result = lcand_valid and in_circle(pt1, pt2, pt3, pt4)
    return result

# Merging functions

def combine_triangulations(ldi, rdi, hull_left, hull_right):
    """
    This function takes two TriangulationEdges class objects and combines
    them into a single TriangulationEdges object.
    Parameters
    ----------
    ldi : TYPE
        DESCRIPTION.
    rdi : TYPE
        DESCRIPTION.
    hull_left : TriangulationEdges
        DESCRIPTION.
    hull_right : TriangulationEdges
        DESCRIPTION.
    Returns
    -------
    base : TYPE
        DESCRIPTION.
    edges : TriangulationEdges
        DESCRIPTION.
    """
    ldo = hull_left.inner
    rdo = hull_right.outer
    rdi += hull_left.num_edges
    rdo += hull_left.num_edges
    
    edges = hull_left.combine_triangulations(hull_right)
    base = edges.connect(edges.edges[ldi].sym, rdi)
    
    # Correct the base edge
    ldi_org = edges.points[edges.edges[ldi].org]
    ldo_org = edges.points[edges.edges[ldo].org]
    rdi_org = edges.points[edges.edges[rdi].org]
    rdo_org = edges.points[edges.edges[rdo].org]
    
    if list_equal(ldi_org, ldo_org):
        ldo = base
    if list_equal(rdi_org, rdo_org):
        rdo = edges.edges[base].sym
    
    edges.set_extreme_edges(ldo, rdo)
    
    return base, edges

def zip_hulls(base, triang):
    """
    Given a triangulation containing two seperate hulls and the base edge 
    connecting the hulls, triangulate the space between the hulls. This is
    refered to as 'zipping' the hulls together.
    
    Parameters
    ----------
    base : int
        index of base edge
    triang : TriangulationEdges 
        Incomplete triangulation, with known base edge
    Returns
    -------
    d_triang : TriangulationEdges 
        Instance of TriangulationEdges class object containing the finished
        Delaunay triangulation of the input triangulation. 
    """
    while True:
        # Make variables for commonly used base edge points
        base1 = triang.points[triang.edges[base].org]
        base2 = triang.points[triang.edges[base].dest]
        
        # Find the first candidate edges for triangulation from each subset
        rcand = triang.edges[triang.edges[base].sym].onext
        pt1 = triang.points[triang.edges[rcand].dest]
        rcand_valid = on_right(base1, base2, pt1)
        
        lcand = triang.edges[base].oprev
        pt2 = triang.points[triang.edges[lcand].dest]
        lcand_valid = on_right(base1, base2, pt2)
        
        # If neither candidate is valid, hull merge is complete
        if not rcand_valid and not lcand_valid:
            break
        
        if rcand_valid:
            triang, rcand = rcand_func(triang, rcand, base1, base2)

        if lcand_valid:
            triang, lcand = lcand_func(triang, lcand, base1, base2)
        
        lcand_strong_valid = candidate_decider(rcand, lcand, lcand_valid, triang)
        
        if not rcand_valid or lcand_strong_valid:
            base = triang.connect(lcand, triang.edges[base].sym)
        else:
            base = triang.connect(triang.edges[base].sym, triang.edges[rcand].sym)
    return triang

def merge_triangulations(groups):
    """
    Each entry of the groups list is a list of two (or one) triangulations. 
    This function takes each pair of triangulations and combines them. 
    Parameters
    ----------
    groups : list
        List of pairs of triangulations
    Returns
    -------
    list
        List of merged triangulations
    """
    triangulations = []
    for group in groups:
        if len(group)==2:
            # Find the first edges to connect the seperate triangulations
            ldi, rdi = lowest_common_tangent(group[0], group[1])
            
            # Combine the two hulls into a single set of edges
            base, d_triang = combine_triangulations(ldi, rdi, group[0], group[1])
            
            # Given the starting base edge, fill in the edges between the hulls
            d_triang = zip_hulls(base, d_triang)
            triangulations.append(d_triang)
        else:
            triangulations.append(group[0])
    return [triangulations[i:i+2] for i in range(0, len(triangulations), 2)]
    
def points_splitter_3(points):
    num = len(points)
    if num%3==0:
        return [points[i:i+3] for i in range(0, num, 3)]
    elif num%3==1:
        split = [points[i:i+2] for i in range(0, 4, 2)]
        split2 = [points[i:i+3] for i in range(4, num, 3)]
    elif num%3==2:
        split = [[points[0], points[1]]]
        split2 = [points[i:i+3] for i in range(2, num, 3)]
    return split + split2

def recursive_group_merge(groups):
    """
    Recursivly call the merge_triangulations() function, until all points have
    been triangulated.
    Parameters
    ----------
    groups : list
        List of pairs of triangulations
    Returns
    -------
    list
        List containing the single completed Delauney triangulation
    """
    while len(groups[0])!=1:
        groups = merge_triangulations(groups)
    return groups

# File Functions

import json

class Input: 
    def __init__(self, world_size, points, workers_num):
        self.world_size = world_size
        self.points = points
        self.workers_num = workers_num

    def desctruct(self):
        return self.world_size, self.points, self.workers_num

    def writeTo(self, filename, cls=json.JSONEncoder):
        with open(filename,'w') as fp:
            json.dump(self.__dict__, fp, cls=cls)

    @staticmethod
    def readFrom(filename):
        with open(filename,'r') as fp:
            jsonObj = json.load(fp=fp)            
            return Input(jsonObj['world_size'], jsonObj['points'], jsonObj['workers_num'])
            
        
class Output:
    def __init__(self, world_size, edges, points, workers_num=0):
        self.world_size = world_size
        self.points = points
        self.edges = edges
        self.workers_num = workers_num
    
    def writeTo(self, filename, cls=json.JSONEncoder):
        with open(filename,'w') as fp:
            json.dump(self.__dict__, fp, cls=cls)
    
    def desctruct(self):
        return self.world_size, self.points, self.edges, self.workers_num

    @staticmethod
    def readFrom(filename):
        with open(filename,'r') as fp:
            jsonObj = json.load(fp=fp)            
            return Output(jsonObj['world_size'], jsonObj['edges'], jsonObj['points'], jsonObj['workers_num'])

    @staticmethod
    def readFromString(str):
        jsonObj = json.loads(s=str)            
        return Output(jsonObj['world_size'], jsonObj['edges'], jsonObj['points'], jsonObj['workers_num'])
            

# Main function

def triangulate(pts_subset):
    """
    This function encapsulates the whole triangulation algorithm into four
    steps. The function takes as input a list of points. Each point is of the 
    form [x, y], where x and y are the coordinates of the point.
    
    Step 1) The list of points is split into groups. Each group has exactly 
            two or three points.
    Step 2) For each group of two point, a single edge is generated. For each
            group of three points, three edges forming a triangle are 
            generated. These are the 'primitive' triangulations. 
    Step 3) The primitive triangulations are paired into groups. 
    Step 4) The groups are then recursively merged until there is only a 
            single triangulation of all points remaining.
    Parameters
    ----------
    pts_subset : list
        A list of points with the form [ [x1, y1], [x2, y2], ..., [xn, yn] ]
        The first element of each list represents the x-coordinate, the second 
        entry the y-coordinate. 
    Returns
    -------
    out : list
        List with a single element. The TriangulationEdges class object with
        the completed Delauney triangulation of the input points. 
        See TriangulationEdges docstring for further info.
    """
    split_pts = groups_of_3(pts_subset)
    primitives = make_primitives(split_pts)
    groups = [primitives[i:i+2] for i in range(0, len(primitives), 2)]
    groups = recursive_group_merge(groups)
    return groups[0][0]

class World():
    """
    Define the range of possible values to generate points for.
    """
    def __init__(self, world_size):
        self.x_min = world_size[0]
        self.x_max = world_size[1]
        self.y_min = world_size[2]
        self.y_max = world_size[3]

world_options = {'min_x_val' : 0,
                 'max_x_val' : 100,
                 'min_y_val' : 0,
                 'max_y_val' : 100}

class TriangulationEncoder(json.JSONEncoder):
    def default(self, obj):
        if not isinstance(obj, Edge) and not isinstance(obj, TriangulationEdges):
            return super(TriangulationEncoder, self).default(obj)

        return obj.__dict__


# Parcs solver 
from Pyro4 import expose
import pickle

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))

        world_size, points, workers_num = self.read_input()

        print('Total points: %d' % len(points))

        ctrl_space_size = min(workers_num,len(self.workers))
        split_pts = points_splitter_3(points)
        pts_per_core = int(len(split_pts)/ctrl_space_size)+1

        print('Points per core: %d' % pts_per_core)
        data = [split_pts[i:i + pts_per_core] for i in range(0, len(split_pts), pts_per_core)] 

        print('Chunks numebr %d' % len(data))
        mapped_opperations = []
        for i in range(0, ctrl_space_size):
            mapped_opperations.append(self.workers[i].build(data[i]))

        mapped = self.myreduce(mapped_opperations)
        
        triangulation = recursive_group_merge(mapped)
        triangulation = triangulation[0][0]

        # output
        self.write_output(triangulation, world_size, ctrl_space_size)

        print("Job Finished")

    @staticmethod
    @expose
    def build(data):
        primitives = make_primitives(data)
        groups = [primitives[i:i+2] for i in range(0, len(primitives), 2)]
        triangulation = recursive_group_merge(groups)
        return pickle.dumps(triangulation[0][0])

    @staticmethod
    def myreduce(mappedOpperations):
        output = []
        for i in range(0, len(mappedOpperations), 2):
            group = [pickle.loads(mappedOpperations[i].value),pickle.loads(mappedOpperations[i+1].value)]
            output.append(group)
        return output

    def read_input(self):
        world_size, points, workers_num = Input.readFrom(filename=self.input_file_name).desctruct()
        return world_size, lexigraphic_sort(points), workers_num

    def write_output(self, output, world_size, workers_num=0):
        Output(world_size, output.edges, output.points, workers_num).writeTo(filename=self.output_file_name, cls=TriangulationEncoder)