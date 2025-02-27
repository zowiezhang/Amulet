from config import args
from responser import Responser


if __name__ == '__main__':
    
    if not args.multi_gpu:
	    if args.method == 'beam' and args.num_beams > 4:
	        args.batch_size = 2
	    elif args.method in ['la', 'amulet']:
	        args.batch_size = 4
	    else:
	        args.batch_size = 8

    responser = Responser(args)
    print("Preference:", args.pref_name)
    
    responser.get_response(args.eval_data, args.pref_name)












