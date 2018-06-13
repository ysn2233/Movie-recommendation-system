from usercf import UserCF
from itemcf import ItemCF
from avg_baseline import evaluate_baselines



def evaluate_cfs():
    ucf = UserCF()
    icf = ItemCF()
    rmse_movie_avg, rmse_user_avg = evaluate_baselines()
    rmse_user_cf = ucf.evaluate()
    rmse_item_cf = icf.evaluate()

    print('RMSE of rs algorithms:')
    print('Movie-average baselines: {}'.format(rmse_movie_avg))
    print('User-average baselines: {}'.format(rmse_user_avg))
    print('UserCF: {}'.format(rmse_user_cf))
    print('ItemCF: {}'.format(rmse_item_cf))


evaluate_cfs()