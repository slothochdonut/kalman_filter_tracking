#simluate trajectory
function sample_trajectory(x, T, β, Q, μ, Σ, F, H, B=zeros(size(F)), u=zeros(size(F, 2)))
    s0 = [x]
    for t in 1:T-1
        x = F * x + B *u + rand(MultivariateNormal(β, Q))
        push!(s0, x)
    end
    s = [copy(m) for m in s0]
    #s[1] = [5.0, 5.0]
    for t in 2:T
        s[t] = H * s[t] + rand(MultivariateNormal(μ, Σ))
    end
    return (s0, s)
end


# tracking function
function tracking(obs, img, h) 
    i, j = round.(Int, obs)
    c = CartesianIndices((i-h:i+h, j-h:j+h)) #create a square of size 2h*2h, center at (i, j)
    x_est = sum((img[k]*k[1]) for k in c)/sum(img[c]) #estimation of x-coordinate
    y_est = sum((img[k]*k[2]) for k in c)/sum(img[c]) #estimation of y-coordinate
    #x_est, y_est = myfindmax(img[k] for k in c) not stable
    box = collect(Iterators.product([i-h,i+h], [j-h,j+h]))
    return (x_est, y_est), box
end


#read the selected interval of video
function read_video((start_sec, end_sec)::Tuple, file_path)
    f = VideoIO.openvideo(file_path)
    seek(f, start_sec)     #read from second = start_sec
    Img = read(f)
    Imgs = [rotr90(Matrix(Gray.(restrict(Img))))] #The size of restricted img is approximately 1/2 of the original size.
    i = 1
    dur = (end_sec-start_sec)*50
    while !eof(f) && i<dur
        i += 1
        read!(f, Img)
        push!(Imgs, rotr90(Matrix(Gray.(restrict(Img)))))
    end
    close(f)
    n = length(Imgs)

    return n, Imgs
end

#finding start point
function start_point(imgs, k=5, idx =2)
    a, b = size(imgs[1])  #size of the image
    K = 2*k
    inner = zeros(a, b)  #initiate as zero matrices
    outer = zeros(a, b)
    for c in CartesianIndices((K+1:a-K-1, K+1:b-K-1))
        i, j = (c[1], c[2])
        inner_ = norm(imgs[idx][i-k:i+k, j-k:j+k] - imgs[idx-1][i-k:i+k, j-k:j+k])
        outer_ = norm(imgs[idx][i-K:i+K, j-K:j+K] - imgs[idx-1][i-K:i+K, j-K:j+K])
        inner[i, j] = inner_
        outer[i, j] = outer_
    end

"""    
for c in CartesianIndices((1:a, 1:b))
        i, j = (c[1], c[2])
        inner_ = norm(imgs[idx][i-k:i+k, j-k:j+k] - imgs[idx-1][i-k:i+k, j-k:j+k])
        outer_ = norm(imgs[idx][i-K:i+K, j-K:j+K] - imgs[idx-1][i-K:i+K, j-K:j+K])
        inner[i, j] = inner_
        outer[i, j] = outer_
    end

"""

    #the second element of findmax() is coordinates
    point = findmax((outer .- inner .< 0.06) .*(outer .> 0.1) .* max.(inner .- 0.08, 0))[2]
    return point, inner, outer
end


# kalman filter #
function predict(x, F, P, Q, B=zeros(size(F)), u=zeros(size(F, 2)))
    x = F*x + B*u #predicted x 
    P = F*P*F' + Q  #predicted error covariance (describes the squared uncertainty of predicted x)
    x, P  
end

function correct(x, y, Ppred, R, H)  # y is the new observation input
    yres = y - H*x # innovation residual     //H*x as prediction of y

    S = (H*Ppred*H' + R) # innovation covariance    //similar to P in predict step 

    K = Ppred*H'/S # Kalman gain    
    x = x + K*yres
    #P = (I - K*H)*Ppred*(I - K*H)' + K*R*K' #  Joseph form
    P = (I - K*H)*Ppred  #--> Ppred - K*H*Ppred   // corrected P, P converges to a fixed value
    x, P, yres, S
end
#use the analytical solution of kalman filter 

"""
     track(A::Matrix{Float64}, p, h = 10) -> p2, err
 Track blurry lightsource by applying a window with half-width `h` at
 an approximate location `p = (i,j)` and find the
 average weighted location of points with *high* light intensity.
 Gives an standard error estimate.
 CR is the window.  taking intersection in case the window on the edge of image
 """
function track(img::Matrix, loc, h = 10)  #loc is the given approximate location
    i, j = round.(Int, loc)
    m, n = size(img)
    CR = intersect(CartesianIndices((1:m, 1:n)), CartesianIndices((i-h:i+h, j-h:j+h)))
    μ = mean(img[ci] for ci in CR)
    C = sum(max(img[ci] - μ, 0) for ci in CR)

    xhat = sum(max(img[ci] - μ, 0)*ci[1] for ci in CR)/C  #ci is each pixel in the window
    yhat = sum(max(img[ci] - μ, 0)*ci[2] for ci in CR)/C  

    xerr =  sum(max(img[ci] - μ, 0)*(ci[1] - xhat)^2 for ci in CR)/C
    yerr =  sum(max(img[ci] - μ, 0)*(ci[2] - yhat)^2 for ci in CR)/C

    err = sqrt(xerr + yerr)
    SVector(xhat, yhat), err
 end
#plot err to see when the beetle disappear 