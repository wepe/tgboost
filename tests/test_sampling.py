from tgboost import sampling

rs = sampling.RowSampler(10000, 0.7)
print type(rs.row_mask), rs.row_mask.sum(), rs.row_mask
rs.shuffle()
print rs.row_mask, rs.row_mask.sum()
rs.shuffle()
print rs.row_mask


cs = sampling.ColumnSampler(100,0.7)
print type(cs.col_selected), cs.col_selected
cs.shuffle()
print cs.col_selected
