"""This module is for building option manager in a modular and generic fashion.

The module includes functionality for: 
    
    -defining options: 
        1. Create a yaml file with a dictionary containing your option 
            definition in the following format:

            {
                options:
                    option_block_1:
                        option_1:
                            #add here positional arguments for add_argparse as list
                            args:
                                ['--option_1',...]
                            #add here keyword arguments for add_argparse in dict structure
                            kwargs:
                                default: 10
                                #type types as strings, they are converted with 
                                #pydoc.locate() method
                                type: 'int' 
                                ...
                        option_2: 
                            ...
                    option_block_2:
                        ...
            }

        2. Pass the option definition file to OptManager by placing the 
            file at default location (<root_folder>/opt_def.yaml) or specify
            a different path when instantiating the option manager or via the
            command line parameter --opt_def <path>.
    parsing options: Defined options can be set by a config file and/or directly
        from command line arguments. Note that options set by the config file
        are overwritten by options set directly via command line arguments. The
        config file can be located at default location 
        (<root_folder>/config.yaml) or a different path can be specified via the
        command line parameter --config <path>. The config file must be a yaml
        file containing a dictionary with the options in the following format:
        {
            options: 
                option_block_1: 
                    option_1: value_1,
                    option_2: value_2,
                    ...
                option_block_2:
                    ...
        }

"""

from pathlib import Path
import yaml
import argparse
from pydoc import locate
import copy


class OptManager:
    def __init__(self,
        config_default=Path() / 'config.yaml',
        opt_def_default=Path() / 'opt_def.yaml',
        default_flow_style=None
        ):
        """Constructor for OptManager.

        Args:
            config_default (obj): Default path (path object of pathlib) to
                config file. Used when not overwritten by command line
                parameter.
            opt_def_default (obj): Default path (path object of pathlib) to 
                yaml file containing option defintion. Used when not
                overwritten by command line parameter.
            default_flow_style (bool): Setting for dumping yaml files. None
                means automatically chosing inline and block style. False 
                means block style and True inline style.
        """

        self.default_flow_style = default_flow_style
        self.block_structure = {'options':{}}

        self.conf_parser = argparse.ArgumentParser(
                add_help=False
                )

        self.conf_parser.add_argument(
                '-c', '--config', 
                default = str(config_default),
                help='Specify options with config file.', 
                metavar='FILE')

        self.conf_parser.add_argument(
                '--opt_def', 
                default=str(opt_def_default), 
                help='Specify yaml file for option definition.',
                metavar='FILE'
                )

        self.parser = argparse.ArgumentParser(
                parents = [self.conf_parser],    
                formatter_class=argparse.RawDescriptionHelpFormatter,
                )

        self.opt_def_dict = None

    def _def_opts_by_dict(self, opt_def_dict):
        for block, block_opts in opt_def_dict['options'].items():
            block_parser = self.parser.add_argument_group(block)
            if not block in self.block_structure:
                self.block_structure['options'][block] = {}
            for opt, arg_types in block_opts.items():
                args = arg_types['args']
                kwargs = arg_types['kwargs']
                if 'type' in kwargs:
                    if isinstance(kwargs['type'], str):
                        type_str = kwargs['type']
                        kwargs['type'] = locate(type_str)
                block_parser.add_argument(*args, **kwargs)
                self.block_structure['options'][block][opt] = 0

    def _def_opts(self, opt_def_yaml): 
        self.opt_def_dict = OptManager.yaml2dict(opt_def_yaml)
        self._def_opts_by_dict(self.opt_def_dict)

    def parse(self, args=None, ignore_remaining=False):
        opts = None
        remaining_opts = None
        if args is None:
            opts, remaining_opts = self.conf_parser.parse_known_args()
        else:
            opts, remaining_opts = self.conf_parser.parse_known_args(args)

        opt_def = Path(opts.opt_def)
        self._def_opts(opt_def)

        defaults = {}
        opts_conf = {}
        if opts.config:
            config = Path(opts.config)
            opts_set_conf = OptManager.yaml2dict(config)
            for blocks, block_opts in opts_set_conf['options'].items():
                for opt, value in block_opts.items():
                    opts_conf[opt] = value
        self.parser.set_defaults(**opts_conf)
        if ignore_remaining:
            opts, _ = self.parser.parse_known_args([])
            if remaining_opts:
                pass
                # print("THERE ARE REMAINING OPTS!!!!")
                # print(remaining_opts)
                # print()
        else:
            opts = self.parser.parse_args(remaining_opts)

        return opts

    def obj2dict(self, opts):
        opt_set = copy.deepcopy(self.block_structure)
        opts = vars(opts)
        for block, block_opts in opt_set['options'].items():
            for opt in block_opts.keys():
                if opt in opts.keys():
                    opt_set['options'][block][opt] = opts[opt]
        return opt_set

    def obj2yaml(self, opts, dest):
        opt_set = self.obj2dict(opts)
        OptManager.dict2yaml(opt_set, dest, self.default_flow_style)

    def opt_def2yaml(self, dest):
        self.dict2yaml(self.opt_def_dict, dest,
            default_flow_style=self.default_flow_style
        )

    @staticmethod
    def agg_opt_blocks(*opt_blocks, dest, default_flow_style=None):
        opt_def = {}
        for opt_block in opt_blocks:
            if opt_def=={}:
                opt_def = OptManager.yaml2dict(opt_block)
            else:
                opt_block_def = OptManager.yaml2dict(opt_block)
                opt_def["options"].update(opt_block_def["options"])

        OptManager.dict2yaml(opt_def, dest, default_flow_style)

        return opt_def

    @staticmethod
    def dict2yaml(some_dict, dest, default_flow_style=None):
        with dest.open('w') as f:
            yaml.dump(some_dict, f, default_flow_style=default_flow_style)

    @staticmethod
    def yaml2dict(source):
        with source.open('r') as f:
            loaded_dict = yaml.load(f, Loader=yaml.Loader)

        return loaded_dict
