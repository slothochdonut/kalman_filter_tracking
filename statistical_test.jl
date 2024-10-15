using Distributions, Statistics, LinearAlgebra
using StaticArrays
using GLMakie; gl = GLMakie
using DelimitedFiles

X = readdlm("trajectories.csv", ',')
V = readdlm("velocity.csv", ',')

burrow = SVector(786, 304)
#ΔX = diff(X, dims =1)
ΔV = diff(V, dims =1)
centered_X = X .- burrow'

X_norm = norm.(eachrow(centered_X))
ΔV_norm = norm.(eachrow(ΔV))
V_norm = norm.(eachrow(V))


scatter(X_norm)
std(X_norm[1050:4600])
std(X_norm[4600:end])
d = dot.(eachrow(centered_X ./X_norm), eachrow(V./V_norm))
scatter(d[1050:end], color=X_norm[1050:end])

A = hcat(ones(length(X_norm[1050:4600])), X_norm[1050:4600])
y = d[1050:4600]

θ = A\y

